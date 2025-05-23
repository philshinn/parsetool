#!/usr/bin/env python3

import argparse
import xml.etree.ElementTree as ET
import sys
import os
import re # For parsing repeat attributes
import random # For generation

# Define the GRXML namespace
GRXML_NS = "http://www.w3.org/2001/06/grammar"
XML_NS = "http://www.w3.org/XML/1998/namespace" # For xml:lang, xml:base

# --- Configuration for Generation ---
DEFAULT_GEN_CONFIG = {
    "max_additional_unbounded_repeats": 2, # For * (0-) and + (1-), repeat min + (0 to this many) times
    "max_absolute_repeats": 5,           # Hard cap for any repeat count (e.g., if grammar says repeat="1-100")
    "max_recursion_depth": 25,           # To prevent infinite loops in recursive grammars
    "void_produces_failure": True,       # If True, a VOID ruleref makes generation fail for that path
    "garbage_placeholder": "[GARBAGE]",  # What to generate for <ruleref special="GARBAGE"/>
    "max_single_sentence_attempts": 10   # How many times to try generating one sentence if previous attempts fail
}
# --- End Generation Configuration ---

class GrxmlValidationError(Exception):
    pass

class GrxmlGenerationError(Exception):
    pass

class GrxmlParser:
    def __init__(self, grxml_filepath, generation_config=None):
        self.filepath = grxml_filepath
        self.tree = None
        self.root_grammar_element = None
        self.rules = {}  # Stores rule_id -> rule_element
        self.root_rule_id = None
        
        self.gen_config = DEFAULT_GEN_CONFIG.copy()
        if generation_config:
            self.gen_config.update(generation_config)
            
        self._validate_and_load()

    def _validate_and_load(self):
        # ... (validation code from previous version - no changes here) ...
        if not os.path.exists(self.filepath):
            raise GrxmlValidationError(f"Error: File not found: {self.filepath}")
        if not self.filepath.lower().endswith(".grxml"):
            raise GrxmlValidationError(f"Error: File extension is not .grxml: {self.filepath}")

        try:
            self.tree = ET.parse(self.filepath)
        except ET.ParseError as e:
            raise GrxmlValidationError(f"Error: Invalid XML in {self.filepath}: {e}")

        self.root_grammar_element = self.tree.getroot()
        if self.root_grammar_element.tag != f"{{{GRXML_NS}}}grammar":
            raise GrxmlValidationError(
                f"Error: Root element is not <grammar> with namespace {GRXML_NS}. Found: {self.root_grammar_element.tag}"
            )

        version = self.root_grammar_element.get("version")
        if version != "1.0":
            print(f"Warning: Grammar version is '{version}', expected '1.0'. Proceeding anyway.", file=sys.stderr)

        self.root_rule_id = self.root_grammar_element.get("root")
        if not self.root_rule_id:
            raise GrxmlValidationError("Error: <grammar> element must have a 'root' attribute.")

        ns_map = {'gr': GRXML_NS}
        found_root_rule = False
        for rule_element in self.root_grammar_element.findall("gr:rule", ns_map):
            rule_id = rule_element.get("id")
            if not rule_id:
                raise GrxmlValidationError("Error: Found <rule> element without an 'id' attribute.")
            if rule_id in self.rules:
                raise GrxmlValidationError(f"Error: Duplicate rule id: {rule_id}")
            self.rules[rule_id] = rule_element
            if rule_id == self.root_rule_id:
                found_root_rule = True
        
        if not found_root_rule:
            raise GrxmlValidationError(f"Error: Root rule '{self.root_rule_id}' not defined.")
        print(f"Successfully validated GRXML structure: {self.filepath}", file=sys.stderr)


    def _parse_repeat_attr(self, repeat_str):
        # ... (code from previous version - no changes here) ...
        if not repeat_str:
            return 1, 1
        
        if repeat_str.isdigit():
            val = int(repeat_str)
            return val, val
        
        if repeat_str == "optional": return 0, 1
        if repeat_str == "plus": return 1, float('inf')
        if repeat_str == "star": return 0, float('inf')

        if repeat_str.endswith("-"):
            try:
                min_r = int(repeat_str[:-1])
                return min_r, float('inf')
            except ValueError:
                raise GrxmlValidationError(f"Invalid repeat attribute format: {repeat_str}")


        match = re.match(r"(\d+)-(\d+)", repeat_str)
        if match:
            min_r, max_r = int(match.group(1)), int(match.group(2))
            if min_r > max_r:
                raise GrxmlValidationError(f"Invalid repeat range: min > max in {repeat_str}")
            return min_r, max_r
        
        # Check for simple "0-1" which is common
        if repeat_str == "0-1":
            return 0, 1
            
        raise GrxmlValidationError(f"Unsupported repeat attribute format: {repeat_str}")

    # --- String Parsing Methods (from previous version, largely unchanged) ---
    def _match_element(self, element, tokens):
        # ... (code from previous version) ...
        tag_local_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag
        repeat_str = element.get('repeat', '1')
        try:
            min_repeats, max_repeats = self._parse_repeat_attr(repeat_str)
        except GrxmlValidationError as e: # Catch parsing errors for repeat during matching
            print(f"Warning: {e} for element {ET.tostring(element, encoding='unicode')[:50]}... Assuming repeat='1'.", file=sys.stderr)
            min_repeats, max_repeats = 1,1


        current_iteration_token_states = [tokens]
        all_successful_outcomes = []

        if min_repeats == 0:
            all_successful_outcomes.append(tokens) 

        # Cap infinite for practical reasons during matching
        # Max_repeats for matching can be higher than for generation
        effective_max_match_repeats = max_repeats
        if max_repeats == float('inf'):
            effective_max_match_repeats = len(tokens) + 1 # Heuristic: can't repeat more times than tokens exist + 1 for empty match
            if min_repeats > 0: effective_max_match_repeats = max(min_repeats, effective_max_match_repeats)


        for i in range(1, int(effective_max_match_repeats) + 1):
            if not current_iteration_token_states: 
                break
            
            next_iteration_token_states = []
            for prev_tokens in current_iteration_token_states:
                content_match_results = []
                if tag_local_name == "item":
                    content_match_results = self._match_item_content(element, prev_tokens)
                elif tag_local_name == "one-of":
                    content_match_results = self._match_one_of_content(element, prev_tokens)
                elif tag_local_name == "ruleref":
                    content_match_results = self._match_ruleref_content(element, prev_tokens)
                elif tag_local_name == "token": 
                    text_to_match = (element.text or "").strip().lower()
                    if prev_tokens and text_to_match and prev_tokens[0] == text_to_match: # Ensure text_to_match is not empty
                        content_match_results.append(prev_tokens[1:])
                    elif not text_to_match: # Empty token matches empty
                         content_match_results.append(prev_tokens)
                elif tag_local_name == "tag":
                    content_match_results.append(prev_tokens) 
                else:
                    content_match_results.append(prev_tokens)


                next_iteration_token_states.extend(content_match_results)
            
            current_iteration_token_states = next_iteration_token_states
            
            if i >= min_repeats:
                all_successful_outcomes.extend(current_iteration_token_states)

            if max_repeats == float('inf') and not current_iteration_token_states: 
                break
        
        unique_outcomes_tuples = {tuple(outcome) for outcome in all_successful_outcomes}
        return [list(outcome_tuple) for outcome_tuple in unique_outcomes_tuples]

    def _match_item_content(self, item_element, tokens):
        # ... (code from previous version) ...
        item_text = (item_element.text or "").strip().lower()
        children = list(item_element)
        current_token_states = [tokens]

        if item_text:
            next_token_states = []
            item_text_tokens = [t for t in item_text.split() if t] 
            if not item_text_tokens:
                 next_token_states.extend(current_token_states)
            else:
                for state in current_token_states:
                    if len(state) >= len(item_text_tokens) and \
                       state[:len(item_text_tokens)] == item_text_tokens:
                        next_token_states.append(state[len(item_text_tokens):])
            current_token_states = next_token_states
            if not current_token_states: 
                return []

        if children:
            return self._match_sequence(children, current_token_states)
        else: 
            return current_token_states

    def _match_one_of_content(self, one_of_element, tokens):
        # ... (code from previous version) ...
        all_possible_outcomes = []
        ns_map = {'gr': GRXML_NS}
        for item_child in one_of_element.findall("gr:item", ns_map):
            results = self._match_element(item_child, tokens) 
            all_possible_outcomes.extend(results)
        
        unique_outcomes_tuples = {tuple(outcome) for outcome in all_possible_outcomes}
        return [list(outcome_tuple) for outcome_tuple in unique_outcomes_tuples]


    def _match_ruleref_content(self, ruleref_element, tokens):
        # ... (code from previous version) ...
        uri = ruleref_element.get("uri")
        special = ruleref_element.get("special")

        if not uri and not special:
            raise GrxmlValidationError("Error: <ruleref> must have 'uri' or 'special' attribute for matching.")

        if special:
            if special.upper() == "NULL": 
                return [tokens]
            elif special.upper() == "VOID": 
                return []
            elif special.upper() == "GARBAGE":
                # Simplified GARBAGE: consumes zero or one token if available
                results = [tokens] 
                if tokens:
                    results.append(tokens[1:]) 
                return results
            else:
                raise GrxmlValidationError(f"Unsupported special ruleref: {special}")


        if uri.startswith("#"):
            rule_id = uri[1:]
            if rule_id not in self.rules:
                raise GrxmlValidationError(f"Error: Undefined rule referenced for matching: {rule_id}")
            
            target_rule_element = self.rules[rule_id]
            return self._match_sequence(list(target_rule_element), tokens)
        else:
            raise GrxmlValidationError(f"Error: External URI or non-local ruleref '{uri}' not supported for matching.")

    def _match_sequence(self, elements, initial_tokens_list_or_single):
        # ... (code from previous version) ...
        if not isinstance(initial_tokens_list_or_single, list):
            raise TypeError("initial_tokens_list_or_single must be a list")

        if not elements: 
             if initial_tokens_list_or_single and isinstance(initial_tokens_list_or_single[0], list):
                 return initial_tokens_list_or_single
             else: 
                 return [initial_tokens_list_or_single]

        if not initial_tokens_list_or_single or not isinstance(initial_tokens_list_or_single[0], list):
             active_states = [initial_tokens_list_or_single] 
        else:
             active_states = initial_tokens_list_or_single

        for element in elements:
            next_active_states = []
            for current_tokens_state in active_states:
                results_for_this_element = self._match_element(element, current_tokens_state)
                next_active_states.extend(results_for_this_element)
            
            active_states = next_active_states
            if not active_states: 
                return []
        
        return active_states

    def parse_string(self, input_string):
        # ... (code from previous version) ...
        if not self.root_rule_id or self.root_rule_id not in self.rules:
            print(f"Error: Root rule '{self.root_rule_id}' is not defined or not found for parsing.", file=sys.stderr)
            return False

        tokens = [t for t in input_string.strip().lower().split() if t]
        root_rule_element = self.rules[self.root_rule_id]
        possible_outcomes = self._match_sequence(list(root_rule_element), tokens)

        for remaining_tokens in possible_outcomes:
            if not remaining_tokens:
                return True
        return False

    # --- Generation Methods ---
    def _determine_repetitions(self, min_reps, max_reps):
        """Determine a random number of repetitions based on min, max, and config."""
        eff_min = min_reps
        eff_max = max_reps

        if max_reps == float('inf'):
            eff_max = min_reps + self.gen_config["max_additional_unbounded_repeats"]
        
        eff_max = min(eff_max, self.gen_config["max_absolute_repeats"])

        # Ensure eff_max is not less than eff_min after caps
        if eff_min > eff_max : # This can happen if min_reps itself is > max_absolute_repeats
            # In this case, we prioritize satisfying min_reps up to a reasonable limit if it's too large
            # Or if config makes eff_max < min_reps (e.g. min_reps=3, max_add_unbounded=0 -> eff_max=3)
            # This path indicates a choice needs to be made. Let's prefer min_reps if it's not overly large.
            if min_reps <= self.gen_config["max_absolute_repeats"]:
                 eff_max = eff_min # If min_reps is acceptable, repeat exactly min_reps times
            else: # min_reps is too large, cap it.
                 print(f"Warning: min_reps ({min_reps}) for generation exceeds max_absolute_repeats ({self.gen_config['max_absolute_repeats']}). Capping at max_absolute_repeats.", file=sys.stderr)
                 eff_min = self.gen_config["max_absolute_repeats"]
                 eff_max = self.gen_config["max_absolute_repeats"]
        
        if eff_min > eff_max: # Final sanity check if logic above had issues.
            # This should ideally not be reached if logic is correct.
            # Default to min_reps if something strange happened.
            return eff_min

        return random.randint(eff_min, int(eff_max))


    def _generate_random_expansion(self, element, depth):
        """
        Generates a list of words by randomly expanding the given element.
        Returns a list of words, or None if generation fails (e.g., VOID encountered).
        """
        if depth > self.gen_config["max_recursion_depth"]:
            print(f"Warning: Max recursion depth ({self.gen_config['max_recursion_depth']}) reached during generation. Path terminated.", file=sys.stderr)
            return ["[RECURSION_LIMIT]"] # Or return None to indicate failure

        tag_local_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag
        
        repeat_str = element.get('repeat', '1')
        try:
            min_reps, max_reps = self._parse_repeat_attr(repeat_str)
        except GrxmlValidationError as e:
             print(f"Warning: Invalid repeat attribute '{repeat_str}' during generation. Assuming '1'. Error: {e}", file=sys.stderr)
             min_reps, max_reps = 1,1

        num_chosen_reps = self._determine_repetitions(min_reps, max_reps)

        if num_chosen_reps == 0:
            return []

        all_generated_words = []
        for _ in range(num_chosen_reps):
            # _expand_element_content_once processes the element's definition for a single pass
            # It does not re-evaluate the repeat attribute of 'element' itself.
            words_from_one_expansion = self._expand_element_content_once(element, depth + 1)
            
            if words_from_one_expansion is None: # A VOID or other failure occurred
                return None 
            all_generated_words.extend(words_from_one_expansion)
        
        return all_generated_words

    def _expand_element_content_once(self, element_def, depth):
        """
        Expands the content of an element definition once.
        This is called by _generate_random_expansion within its repetition loop.
        Returns a list of words, or None on failure.
        """
        tag_local_name = element_def.tag.split('}')[-1] if '}' in element_def.tag else element_def.tag
        generated_words = []
        ns_map = {'gr': GRXML_NS}

        if tag_local_name == "item":
            item_text = (element_def.text or "").strip()
            if item_text:
                generated_words.extend(item_text.split()) # Tokenize by space
            
            # Children of <item> form a sequence
            for child_element in list(element_def):
                child_expansion = self._generate_random_expansion(child_element, depth) # depth+1 was in previous call
                if child_expansion is None:
                    return None
                generated_words.extend(child_expansion)

        elif tag_local_name == "one-of":
            item_choices = element_def.findall("gr:item", ns_map)
            if not item_choices:
                # This is a grammar error, but for generation, treat as ungeneratable path
                print("Warning: <one-of> without <item> children encountered during generation.", file=sys.stderr)
                return None 
            
            chosen_item = random.choice(item_choices)
            # The chosen_item itself will be expanded by _generate_random_expansion, which handles its repeat.
            item_expansion = self._generate_random_expansion(chosen_item, depth) # depth+1 was in previous call
            if item_expansion is None:
                return None
            generated_words.extend(item_expansion)

        elif tag_local_name == "ruleref":
            uri = element_def.get("uri")
            special = element_def.get("special")

            if not uri and not special:
                raise GrxmlGenerationError("Error: <ruleref> must have 'uri' or 'special' attribute for generation.")

            if special:
                special = special.upper()
                if special == "NULL":
                    pass # Generates nothing
                elif special == "VOID":
                    if self.gen_config["void_produces_failure"]:
                        return None
                    # else, VOID generates nothing and allows continuation (less common interpretation)
                elif special == "GARBAGE":
                    placeholder = self.gen_config["garbage_placeholder"]
                    if placeholder: # Only add if placeholder is non-empty
                        generated_words.append(placeholder)
                else:
                    raise GrxmlGenerationError(f"Unsupported special ruleref for generation: {special}")
            
            elif uri.startswith("#"):
                rule_id = uri[1:]
                if rule_id not in self.rules:
                    raise GrxmlGenerationError(f"Error: Undefined rule referenced for generation: {rule_id}")
                
                target_rule_element = self.rules[rule_id]
                # The content of a rule is a sequence of its children elements
                for child_in_rule in list(target_rule_element):
                    rule_child_expansion = self._generate_random_expansion(child_in_rule, depth) # depth+1 was in previous call
                    if rule_child_expansion is None:
                        return None
                    generated_words.extend(rule_child_expansion)
            else:
                raise GrxmlGenerationError(f"Error: External URI or non-local ruleref '{uri}' not supported for generation.")

        elif tag_local_name == "token":
            token_text = (element_def.text or "").strip()
            if token_text:
                generated_words.append(token_text)
        
        elif tag_local_name == "tag":
            # Tags are for semantic interpretation, not typically for spoken/written output.
            # If a tag could contain grammar elements that *should* be expanded (non-standard for SRGS),
            # this would need to iterate list(element_def) like <item>.
            # Assuming standard tags, they generate nothing.
            pass
        
        # else: unknown element, generates nothing for now

        return generated_words


    def generate_sentences(self, num_sentences):
        if not self.root_rule_id or self.root_rule_id not in self.rules:
            raise GrxmlGenerationError(f"Error: Root rule '{self.root_rule_id}' is not defined or not found for generation.")

        root_rule_element = self.rules[self.root_rule_id]
        generated_count = 0
        
        for i in range(num_sentences):
            sentence_words = None
            for attempt in range(self.gen_config["max_single_sentence_attempts"]):
                # Each attempt starts a fresh random walk from the root rule's content
                # The root rule itself doesn't have a repeat attribute; its content is a sequence.
                current_sentence_words = []
                generation_ok = True
                for child_of_root in list(root_rule_element):
                    expansion = self._generate_random_expansion(child_of_root, 0)
                    if expansion is None:
                        generation_ok = False
                        break
                    current_sentence_words.extend(expansion)
                
                if generation_ok:
                    sentence_words = current_sentence_words
                    break # Successful generation for this sentence
                # else, try again (new random choices)
            
            if sentence_words is not None:
                print(" ".join(s for s in sentence_words if s)) # Join non-empty words
                generated_count += 1
            else:
                print(f"Warning: Could not generate sentence {i+1} after {self.gen_config['max_single_sentence_attempts']} attempts. Grammar might be too restrictive or contain unavoidable VOID paths.", file=sys.stderr)
        
        if generated_count < num_sentences:
            print(f"Info: Generated {generated_count} of {num_sentences} requested sentences.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="ParseTool: Validates a .grxml file, parses strings, or generates sentences.")
    parser.add_argument("grxml_file", help="Path to the .grxml speech recognition grammar file.")
    parser.add_argument("-s", "--string", help="A quoted string to parse against the grammar.")
    parser.add_argument("-g", "--generate", type=int, metavar="N", help="Generate N sentences from the grammar.")

    args = parser.parse_args()

    if not args.string and args.generate is None:
        # If only grxml_file is provided, just validate.
        try:
            GrxmlParser(args.grxml_file) # Validation happens in constructor
            print("GRXML file is valid (basic structural check). No string to parse or sentences to generate.", file=sys.stderr)
        except GrxmlValidationError as e:
            print(f"GRXML Validation Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred: {e}", file=sys.stderr)
            sys.exit(1)
        sys.exit(0)

    try:
        # Initialize parser (also validates)
        grammar_parser = GrxmlParser(args.grxml_file)
    except GrxmlValidationError as e:
        print(f"GRXML Validation Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during GRXML loading: {e}", file=sys.stderr)
        sys.exit(1)

    if args.string is not None:
        try:
            print(f"Attempting to parse string: '{args.string}'", file=sys.stderr)
            result = grammar_parser.parse_string(args.string)
            print(result) # Output True or False to stdout
            if not result:
                 sys.exit(2) # Indicate parsing failure
        except GrxmlValidationError as e:
            print(f"Error during string parsing attempt: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred during string parsing: {e}", file=sys.stderr)
            # import traceback; traceback.print_exc() # For debugging
            sys.exit(1)
            
    if args.generate is not None:
        if args.generate <= 0:
            print("Error: Number of sentences to generate (-g N) must be positive.", file=sys.stderr)
            sys.exit(1)
        try:
            print(f"Attempting to generate {args.generate} sentences...", file=sys.stderr)
            grammar_parser.generate_sentences(args.generate)
        except GrxmlGenerationError as e:
            print(f"Error during sentence generation: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred during sentence generation: {e}", file=sys.stderr)
            # import traceback; traceback.print_exc() # For debugging
            sys.exit(1)

if __name__ == "__main__":
    main()