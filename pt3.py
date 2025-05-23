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

DEFAULT_GEN_CONFIG = {
    "max_additional_unbounded_repeats": 2,
    "max_absolute_repeats": 5,
    "max_recursion_depth": 25,
    "void_produces_failure": True,
    "garbage_placeholder": "[GARBAGE]",
    "max_single_sentence_attempts": 10
}

class GrxmlValidationError(Exception):
    pass

class GrxmlGenerationError(Exception):
    pass

class GrxmlParser:
    _loaded_grammars_cache = {} # Class variable to cache loaded grammar parsers

    def __init__(self, grxml_filepath, generation_config=None, is_external_load=False):
        # Ensure grxml_filepath is absolute for consistent caching keys
        # If it's an external load, it should already be resolved by the caller.
        # If it's a top-level load, resolve it here.
        if not os.path.isabs(grxml_filepath) and not is_external_load: # Top-level loading
            self.filepath = os.path.abspath(grxml_filepath)
        else: # Already resolved or absolute path provided
            self.filepath = os.path.normpath(grxml_filepath)

        self.tree = None
        self.root_grammar_element = None
        self.rules = {}
        self.root_rule_id = None
        
        self.gen_config = DEFAULT_GEN_CONFIG.copy()
        if generation_config: # This could be from main call or propagated from parent
            self.gen_config.update(generation_config)
            
        self._validate_and_load()

        # Add to cache after successful load, if not already (e.g. direct instantiation for top level)
        # The _get_external_grammar method will primarily manage caching for externals.
        if self.filepath not in GrxmlParser._loaded_grammars_cache:
            GrxmlParser._loaded_grammars_cache[self.filepath] = self


    def _get_external_grammar(self, external_uri_path_str):
        """
        Resolves, loads (if necessary), and returns a GrxmlParser instance for an external grammar.
        external_uri_path_str is the file path part of a ruleref URI.
        """
        current_grammar_dir = os.path.dirname(self.filepath)
        
        if os.path.isabs(external_uri_path_str):
            resolved_external_path = os.path.normpath(external_uri_path_str)
        else:
            resolved_external_path = os.path.normpath(os.path.join(current_grammar_dir, external_uri_path_str))

        if not os.path.exists(resolved_external_path):
            raise GrxmlValidationError(
                f"Error: External grammar file not found: {resolved_external_path} "
                f"(referenced from {self.filepath} via URI part '{external_uri_path_str}')"
            )

        if resolved_external_path in GrxmlParser._loaded_grammars_cache:
            return GrxmlParser._loaded_grammars_cache[resolved_external_path]
        else:
            print(f"Loading external grammar: {resolved_external_path} (referenced from {self.filepath})", file=sys.stderr)
            # Propagate the generation config from the *current* (referencing) grammar.
            external_parser = GrxmlParser(resolved_external_path, 
                                          generation_config=self.gen_config, 
                                          is_external_load=True)
            # The constructor GrxmlParser will add itself to the cache.
            return external_parser

    def _validate_and_load(self):
        # Uses self.filepath, which is now absolute
        if not os.path.exists(self.filepath): # Should be caught by _get_external_grammar if external
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
                f"Error: Root element in {self.filepath} is not <grammar> with namespace {GRXML_NS}. Found: {self.root_grammar_element.tag}"
            )

        version = self.root_grammar_element.get("version")
        if version != "1.0":
            print(f"Warning: Grammar version in {self.filepath} is '{version}', expected '1.0'. Proceeding anyway.", file=sys.stderr)

        self.root_rule_id = self.root_grammar_element.get("root")
        # A grammar doesn't strictly need a root rule if it's only referenced by specific rule IDs externally.
        # However, if referenced by filename only, it needs a root.
        # if not self.root_rule_id:
        #     print(f"Warning: <grammar> element in {self.filepath} has no 'root' attribute.", file=sys.stderr)


        ns_map = {'gr': GRXML_NS}
        # found_root_rule = False # Not strictly necessary for library grammars
        for rule_element in self.root_grammar_element.findall("gr:rule", ns_map):
            rule_id = rule_element.get("id")
            if not rule_id:
                raise GrxmlValidationError(f"Error: Found <rule> in {self.filepath} without an 'id' attribute.")
            if rule_id in self.rules:
                raise GrxmlValidationError(f"Error: Duplicate rule id '{rule_id}' in {self.filepath}")
            self.rules[rule_id] = rule_element
            # if rule_id == self.root_rule_id:
            #     found_root_rule = True
        
        # if self.root_rule_id and not found_root_rule : # Only error if a root is declared but not defined
        #    raise GrxmlValidationError(f"Error: Root rule '{self.root_rule_id}' declared in {self.filepath} but not defined.")
        
        # For the main grammar, a root rule *must* be defined and resolvable if we are to parse/generate from it directly
        # This check is better placed in parse_string/generate_sentences for the *active* grammar.
        
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
        
        if repeat_str == "0-1":
            return 0, 1
            
        raise GrxmlValidationError(f"Unsupported repeat attribute format: {repeat_str}")

    # --- String Parsing Methods ---
    def _match_element(self, element, tokens):
        # ... (largely same as before, ensure it uses self. for recursive calls) ...
        tag_local_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag
        repeat_str = element.get('repeat', '1')
        try:
            min_repeats, max_repeats = self._parse_repeat_attr(repeat_str)
        except GrxmlValidationError as e: 
            print(f"Warning: {e} for element in {self.filepath}. Assuming repeat='1'.", file=sys.stderr)
            min_repeats, max_repeats = 1,1

        current_iteration_token_states = [tokens]
        all_successful_outcomes = []

        if min_repeats == 0:
            all_successful_outcomes.append(tokens) 

        effective_max_match_repeats = max_repeats
        if max_repeats == float('inf'):
            effective_max_match_repeats = len(tokens) + 1 
            if min_repeats > 0: effective_max_match_repeats = max(min_repeats, effective_max_match_repeats)

        for i in range(1, int(effective_max_match_repeats) + 1):
            if not current_iteration_token_states: 
                break
            
            next_iteration_token_states = []
            for prev_tokens in current_iteration_token_states:
                content_match_results = []
                # These calls use `self.` which correctly dispatches to the current grammar's methods
                if tag_local_name == "item":
                    content_match_results = self._match_item_content(element, prev_tokens)
                elif tag_local_name == "one-of":
                    content_match_results = self._match_one_of_content(element, prev_tokens)
                elif tag_local_name == "ruleref": # This will handle local/external
                    content_match_results = self._match_ruleref_content(element, prev_tokens)
                elif tag_local_name == "token": 
                    text_to_match = (element.text or "").strip().lower()
                    if prev_tokens and text_to_match and prev_tokens[0] == text_to_match:
                        content_match_results.append(prev_tokens[1:])
                    elif not text_to_match: 
                         content_match_results.append(prev_tokens)
                elif tag_local_name == "tag":
                    content_match_results.append(prev_tokens) 
                else: # Unknown element, treat as non-consuming in sequence
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
        # ... (same as before, uses self._match_sequence which is fine) ...
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
        # ... (same as before, uses self._match_element which is fine) ...
        all_possible_outcomes = []
        ns_map = {'gr': GRXML_NS}
        for item_child in one_of_element.findall("gr:item", ns_map):
            results = self._match_element(item_child, tokens) 
            all_possible_outcomes.extend(results)
        
        unique_outcomes_tuples = {tuple(outcome) for outcome in all_possible_outcomes}
        return [list(outcome_tuple) for outcome_tuple in unique_outcomes_tuples]

    def _match_ruleref_content(self, ruleref_element, tokens):
        uri = ruleref_element.get("uri")
        special = ruleref_element.get("special")

        if not uri and not special:
            raise GrxmlValidationError(f"Error: <ruleref> in {self.filepath} must have 'uri' or 'special' attribute.")

        if special:
            special_upper = special.upper()
            if special_upper == "NULL": return [tokens]
            if special_upper == "VOID": return []
            if special_upper == "GARBAGE":
                results = [tokens] 
                if tokens: results.append(tokens[1:]) 
                return results
            raise GrxmlValidationError(f"Unsupported special ruleref '{special}' in {self.filepath}")

        if uri.startswith("#"): # Local rule reference
            rule_id = uri[1:]
            if rule_id not in self.rules:
                raise GrxmlValidationError(f"Error: Undefined local rule '{rule_id}' referenced in {self.filepath}")
            target_rule_element = self.rules[rule_id]
            return self._match_sequence(list(target_rule_element), tokens)
        else: # External rule reference
            parts = uri.split('#', 1)
            external_file_path_str = parts[0]
            external_rule_id_frag = parts[1] if len(parts) > 1 else None

            # self._get_external_grammar uses self.filepath to resolve relative paths
            external_grammar_parser = self._get_external_grammar(external_file_path_str)
            
            target_rule_id_in_external = external_rule_id_frag
            if not target_rule_id_in_external: # No fragment, use root rule of external grammar
                target_rule_id_in_external = external_grammar_parser.root_rule_id
                if not target_rule_id_in_external:
                    raise GrxmlValidationError(
                        f"Error: External grammar {external_grammar_parser.filepath} (referenced by {uri} from {self.filepath}) "
                        f"has no root rule specified, and no rule fragment was provided in the URI."
                    )
            
            if target_rule_id_in_external not in external_grammar_parser.rules:
                raise GrxmlValidationError(
                    f"Error: Rule '{target_rule_id_in_external}' not found in external grammar {external_grammar_parser.filepath} "
                    f"(referenced by {uri} from {self.filepath})."
                )
            
            target_rule_element = external_grammar_parser.rules[target_rule_id_in_external]

            # Check scope for non-root specific rule references
            if external_rule_id_frag: # A specific rule was named in the fragment
                scope = target_rule_element.get("scope", "private") # Default to private if not specified
                if scope == "private":
                    raise GrxmlValidationError(
                        f"Error: Attempt to reference private rule '{target_rule_id_in_external}' "
                        f"in external grammar {external_grammar_parser.filepath} (referenced by {uri} from {self.filepath})."
                    )
            
            # Delegate matching to the external_grammar_parser's context
            return external_grammar_parser._match_sequence(list(target_rule_element), tokens)


    def _match_sequence(self, elements, initial_tokens_list_or_single):
        # ... (same as before, uses self._match_element which is fine) ...
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
                # Calls _match_element on the current 'self' instance
                results_for_this_element = self._match_element(element, current_tokens_state)
                next_active_states.extend(results_for_this_element)
            
            active_states = next_active_states
            if not active_states: 
                return []
        return active_states

    def parse_string(self, input_string):
        if not self.root_rule_id or self.root_rule_id not in self.rules:
            raise GrxmlValidationError(f"Error: Root rule '{self.root_rule_id}' for parsing is not defined or not found in {self.filepath}.")

        tokens = [t for t in input_string.strip().lower().split() if t]
        root_rule_element = self.rules[self.root_rule_id]
        possible_outcomes = self._match_sequence(list(root_rule_element), tokens)

        for remaining_tokens in possible_outcomes:
            if not remaining_tokens:
                return True
        return False

    # --- Generation Methods ---
    def _determine_repetitions(self, min_reps, max_reps):
        # ... (same as before) ...
        eff_min = min_reps
        eff_max = max_reps

        if max_reps == float('inf'):
            eff_max = min_reps + self.gen_config["max_additional_unbounded_repeats"]
        
        eff_max = min(eff_max, self.gen_config["max_absolute_repeats"])
        if eff_min > eff_max :
            if min_reps <= self.gen_config["max_absolute_repeats"]:
                 eff_max = eff_min 
            else: 
                 print(f"Warning: min_reps ({min_reps}) for generation exceeds max_absolute_repeats ({self.gen_config['max_absolute_repeats']}). Capping.", file=sys.stderr)
                 eff_min = self.gen_config["max_absolute_repeats"]
                 eff_max = self.gen_config["max_absolute_repeats"]
        if eff_min > eff_max: # Should not happen if logic above is sound
            return eff_min
        return random.randint(eff_min, int(eff_max))


    def _generate_random_expansion(self, element, depth):
        # ... (largely same as before, ensure recursive calls use self.) ...
        if depth > self.gen_config["max_recursion_depth"]:
            print(f"Warning: Max recursion depth ({self.gen_config['max_recursion_depth']}) reached in {self.filepath}. Path terminated.", file=sys.stderr)
            return ["[RECURSION_LIMIT]"] 

        tag_local_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag
        
        repeat_str = element.get('repeat', '1')
        try:
            min_reps, max_reps = self._parse_repeat_attr(repeat_str)
        except GrxmlValidationError as e:
             print(f"Warning: Invalid repeat attribute '{repeat_str}' in {self.filepath} during generation. Assuming '1'. Error: {e}", file=sys.stderr)
             min_reps, max_reps = 1,1

        num_chosen_reps = self._determine_repetitions(min_reps, max_reps)
        if num_chosen_reps == 0: return []

        all_generated_words = []
        for _ in range(num_chosen_reps):
            # Uses self._expand_element_content_once
            words_from_one_expansion = self._expand_element_content_once(element, depth + 1)
            if words_from_one_expansion is None: return None 
            all_generated_words.extend(words_from_one_expansion)
        return all_generated_words

    def _expand_element_content_once(self, element_def, depth):
        # This method handles the dispatch for rulerefs to external grammars
        tag_local_name = element_def.tag.split('}')[-1] if '}' in element_def.tag else element_def.tag
        generated_words = []
        ns_map = {'gr': GRXML_NS}

        if tag_local_name == "item":
            item_text = (element_def.text or "").strip()
            if item_text: generated_words.extend(item_text.split())
            for child_element in list(element_def):
                # Uses self._generate_random_expansion
                child_expansion = self._generate_random_expansion(child_element, depth)
                if child_expansion is None: return None
                generated_words.extend(child_expansion)

        elif tag_local_name == "one-of":
            item_choices = element_def.findall("gr:item", ns_map)
            if not item_choices:
                print(f"Warning: <one-of> in {self.filepath} without <item> children encountered during generation.", file=sys.stderr)
                return None 
            chosen_item = random.choice(item_choices)
            item_expansion = self._generate_random_expansion(chosen_item, depth) # Uses self.
            if item_expansion is None: return None
            generated_words.extend(item_expansion)

        elif tag_local_name == "ruleref":
            uri = element_def.get("uri")
            special = element_def.get("special")

            if not uri and not special:
                raise GrxmlGenerationError(f"Error: <ruleref> in {self.filepath} must have 'uri' or 'special' attribute.")

            if special:
                special_upper = special.upper()
                if special_upper == "NULL": pass
                elif special_upper == "VOID":
                    if self.gen_config["void_produces_failure"]: return None
                elif special_upper == "GARBAGE":
                    placeholder = self.gen_config["garbage_placeholder"]
                    if placeholder: generated_words.append(placeholder)
                else:
                    raise GrxmlGenerationError(f"Unsupported special ruleref '{special}' in {self.filepath} for generation.")
            
            elif uri.startswith("#"): # Local rule
                rule_id = uri[1:]
                if rule_id not in self.rules:
                    raise GrxmlGenerationError(f"Error: Undefined local rule '{rule_id}' referenced in {self.filepath} for generation.")
                target_rule_element = self.rules[rule_id]
                # Content of a rule is a sequence of its children
                for child_in_rule in list(target_rule_element):
                    # Uses self._generate_random_expansion
                    rule_child_expansion = self._generate_random_expansion(child_in_rule, depth)
                    if rule_child_expansion is None: return None
                    generated_words.extend(rule_child_expansion)
            else: # External rule reference
                parts = uri.split('#', 1)
                external_file_path_str = parts[0]
                external_rule_id_frag = parts[1] if len(parts) > 1 else None

                external_grammar_parser = self._get_external_grammar(external_file_path_str) # uses self.
                
                target_rule_id_in_external = external_rule_id_frag
                if not target_rule_id_in_external:
                    target_rule_id_in_external = external_grammar_parser.root_rule_id
                    if not target_rule_id_in_external:
                        raise GrxmlGenerationError(
                            f"Error: External grammar {external_grammar_parser.filepath} (referenced by {uri} from {self.filepath}) "
                            f"has no root rule, and no fragment provided."
                        )
                
                if target_rule_id_in_external not in external_grammar_parser.rules:
                    raise GrxmlGenerationError(
                        f"Error: Rule '{target_rule_id_in_external}' not found in external grammar {external_grammar_parser.filepath} "
                        f"(referenced by {uri} from {self.filepath})."
                    )
                
                target_rule_element = external_grammar_parser.rules[target_rule_id_in_external]

                if external_rule_id_frag: # Specific rule named
                    scope = target_rule_element.get("scope", "private")
                    if scope == "private":
                        raise GrxmlGenerationError(
                            f"Error: Attempt to generate from private rule '{target_rule_id_in_external}' "
                            f"in external grammar {external_grammar_parser.filepath} (referenced by {uri} from {self.filepath})."
                        )
                
                # Delegate generation to the external_grammar_parser's context for its rule
                for child_in_external_rule in list(target_rule_element):
                    # IMPORTANT: Call on external_grammar_parser instance
                    expansion = external_grammar_parser._generate_random_expansion(child_in_external_rule, depth)
                    if expansion is None: return None
                    generated_words.extend(expansion)

        elif tag_local_name == "token":
            token_text = (element_def.text or "").strip()
            if token_text: generated_words.append(token_text)
        
        elif tag_local_name == "tag": pass # Tags generate nothing
        # else: unknown element, generates nothing
        return generated_words


    def generate_sentences(self, num_sentences):
        if not self.root_rule_id or self.root_rule_id not in self.rules:
            raise GrxmlGenerationError(f"Error: Root rule '{self.root_rule_id}' for generation is not defined or not found in {self.filepath}.")

        root_rule_element = self.rules[self.root_rule_id]
        generated_count = 0
        
        for i in range(num_sentences):
            sentence_words = None
            for _attempt in range(self.gen_config["max_single_sentence_attempts"]):
                current_sentence_words = []
                generation_ok = True
                for child_of_root in list(root_rule_element):
                    # Uses self._generate_random_expansion
                    expansion = self._generate_random_expansion(child_of_root, 0)
                    if expansion is None:
                        generation_ok = False
                        break
                    current_sentence_words.extend(expansion)
                
                if generation_ok:
                    sentence_words = current_sentence_words
                    break 
            
            if sentence_words is not None:
                print(" ".join(s for s in sentence_words if s))
                generated_count += 1
            else:
                print(f"Warning: Could not generate sentence {i+1} from {self.filepath} after {self.gen_config['max_single_sentence_attempts']} attempts.", file=sys.stderr)
        
        if generated_count < num_sentences:
            print(f"Info: Generated {generated_count} of {num_sentences} requested sentences from {self.filepath}.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="ParseTool: Validates .grxml, parses strings, or generates sentences. Supports external grammar references.")
    parser.add_argument("grxml_file", help="Path to the main .grxml speech recognition grammar file.")
    parser.add_argument("-s", "--string", help="A quoted string to parse against the grammar.")
    parser.add_argument("-g", "--generate", type=int, metavar="N", help="Generate N sentences from the grammar.")

    args = parser.parse_args()

    # Prepare generation config (even if not generating, external grammars might use it if they were pre-loaded somehow)
    # For this tool, gen_config is primarily for the top-level operation.
    active_gen_config = DEFAULT_GEN_CONFIG.copy()
    # If args could modify gen_config, do it here:
    # e.g. active_gen_config["max_recursion_depth"] = args.max_depth_if_added
    
    initial_grammar_path = args.grxml_file

    if not args.string and args.generate is None:
        try:
            # Just validate the main grammar file
            GrxmlParser(initial_grammar_path, generation_config=active_gen_config)
            print(f"GRXML file {initial_grammar_path} is valid (basic structural check). No string to parse or sentences to generate.", file=sys.stderr)
        except GrxmlValidationError as e:
            print(f"GRXML Validation Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred: {e}", file=sys.stderr)
            # import traceback; traceback.print_exc()
            sys.exit(1)
        sys.exit(0)

    try:
        # Load the main grammar. External grammars will be loaded on demand.
        # The GrxmlParser constructor will handle adding to cache.
        # Or, ensure it's cleared for a fresh run if desired: GrxmlParser._loaded_grammars_cache.clear()
        grammar_parser = GrxmlParser(initial_grammar_path, generation_config=active_gen_config)
    except GrxmlValidationError as e:
        print(f"GRXML Validation Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during GRXML loading: {e}", file=sys.stderr)
        # import traceback; traceback.print_exc()
        sys.exit(1)

    if args.string is not None:
        try:
            print(f"Attempting to parse string: '{args.string}' using {grammar_parser.filepath}", file=sys.stderr)
            result = grammar_parser.parse_string(args.string)
            print(result)
            if not result: sys.exit(2)
        except GrxmlValidationError as e:
            print(f"Error during string parsing attempt: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred during string parsing: {e}", file=sys.stderr)
            # import traceback; traceback.print_exc()
            sys.exit(1)
            
    if args.generate is not None:
        if args.generate <= 0:
            print("Error: Number of sentences to generate (-g N) must be positive.", file=sys.stderr)
            sys.exit(1)
        try:
            print(f"Attempting to generate {args.generate} sentences from {grammar_parser.filepath}...", file=sys.stderr)
            grammar_parser.generate_sentences(args.generate)
        except GrxmlGenerationError as e:
            print(f"Error during sentence generation: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred during sentence generation: {e}", file=sys.stderr)
            # import traceback; traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()