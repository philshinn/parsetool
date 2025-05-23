#!/usr/bin/env python3

import argparse
import xml.etree.ElementTree as ET
import sys
import os
import re # For parsing repeat attributes

# Define the GRXML namespace
GRXML_NS = "http://www.w3.org/2001/06/grammar"
XML_NS = "http://www.w3.org/XML/1998/namespace" # For xml:lang, xml:base

class GrxmlValidationError(Exception):
    pass

class GrxmlParser:
    def __init__(self, grxml_filepath):
        self.filepath = grxml_filepath
        self.tree = None
        self.root_grammar_element = None
        self.rules = {}  # Stores rule_id -> rule_element
        self.root_rule_id = None
        self._validate_and_load()

    def _validate_and_load(self):
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
            # The spec says this is fixed, but allow flexibility if not strictly required
            print(f"Warning: Grammar version is '{version}', expected '1.0'. Proceeding anyway.", file=sys.stderr)
            # raise GrxmlValidationError(f"Error: Grammar version is '{version}', expected '1.0'.")

        # Check for root rule declaration
        self.root_rule_id = self.root_grammar_element.get("root")
        if not self.root_rule_id:
            raise GrxmlValidationError("Error: <grammar> element must have a 'root' attribute.")

        # Pre-process and store rules
        # Using a dictionary for namespaces for findall
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

        # Basic validation passed
        print(f"Successfully validated GRXML structure: {self.filepath}", file=sys.stderr)


    def _parse_repeat_attr(self, repeat_str):
        """Parses repeat attribute string (e.g., "0-1", "1", "0-", "2-5")
           Returns (min_repeats, max_repeats) where max_repeats can be float('inf')
        """
        if not repeat_str: # Default is "1"
            return 1, 1
        
        if repeat_str.isdigit():
            val = int(repeat_str)
            return val, val
        
        if repeat_str.endswith("-"): # e.g. "0-", "1-", "N-"
            min_r = int(repeat_str[:-1])
            return min_r, float('inf')

        match = re.match(r"(\d+)-(\d+)", repeat_str)
        if match:
            min_r, max_r = int(match.group(1)), int(match.group(2))
            if min_r > max_r:
                raise GrxmlValidationError(f"Invalid repeat range: {repeat_str}")
            return min_r, max_r
        
        # Specific cases like "0-1" are handled by the regex above if simplified.
        # The spec allows "optional" (0-1), "plus" (1-), "star" (0-).
        # This simplified parser might not cover all textual forms.
        # For simplicity, sticking to numeric forms for now.
        
        raise GrxmlValidationError(f"Unsupported repeat attribute format: {repeat_str}")


    def _match_element(self, element, tokens):
        """
        Attempts to match the given element against the current list of tokens.
        Returns a list of possible remaining token lists after successful matches.
        An empty list means no match.
        """
        tag_local_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag
        
        # Get repeat attribute
        repeat_str = element.get('repeat', '1') # Default is exactly once
        min_repeats, max_repeats = self._parse_repeat_attr(repeat_str)

        # This list will hold lists of tokens remaining after matching the element's content
        # 'current_iteration_token_states' are the token lists *before* attempting the current repetition.
        # Start with the initial state (tokens before any repetition of this element).
        current_iteration_token_states = [tokens]
        
        # 'all_successful_outcomes' accumulates token lists that satisfy a valid number of repetitions.
        all_successful_outcomes = []

        if min_repeats == 0:
            all_successful_outcomes.append(tokens) # 0 repetitions is a valid match

        for i in range(1, int(max_repeats) + 1 if max_repeats != float('inf') else 1000): # Cap infinite for practical reasons
            if not current_iteration_token_states: # Cannot repeat further
                break
            
            next_iteration_token_states = []
            for prev_tokens in current_iteration_token_states:
                # Try to match the element's content *once*
                content_match_results = []
                if tag_local_name == "item":
                    content_match_results = self._match_item_content(element, prev_tokens)
                elif tag_local_name == "one-of":
                    content_match_results = self._match_one_of_content(element, prev_tokens)
                elif tag_local_name == "ruleref":
                    content_match_results = self._match_ruleref_content(element, prev_tokens)
                elif tag_local_name == "token": # SRGS 1.0 spec, section 2.2.1 Token Content
                    text_to_match = (element.text or "").strip().lower()
                    if prev_tokens and prev_tokens[0] == text_to_match:
                        content_match_results.append(prev_tokens[1:])
                elif tag_local_name == "tag":
                    # Tags do not consume input. Their "content" is semantic.
                    # If a tag has structured children (not typical), this would need extension.
                    # For now, assume tags are leaf-like or their content is ignored for matching.
                    content_match_results.append(prev_tokens) 
                else:
                    # Unknown element, or element whose children should be processed as a sequence
                    # (e.g. a <rule> or an <item> with mixed content)
                    # For now, let's assume these are handled by _match_item_content for items.
                    # If an element doesn't have specific logic, it effectively doesn't consume.
                    # This might need refinement for arbitrary structures.
                    # For now, treat unknown tags as non-consuming / passthrough for sequence matching.
                    # print(f"Warning: Encountered unhandled element type '{tag_local_name}' for direct matching, assuming it passes through.", file=sys.stderr)
                    content_match_results.append(prev_tokens)


                next_iteration_token_states.extend(content_match_results)
            
            current_iteration_token_states = next_iteration_token_states
            
            if i >= min_repeats:
                all_successful_outcomes.extend(current_iteration_token_states)

            if max_repeats == float('inf') and not current_iteration_token_states: # For 0-inf or 1-inf, if one iter fails, stop
                break
        
        # Deduplicate outcomes (e.g. if multiple paths lead to same remaining tokens)
        # Convert lists to tuples to make them hashable for set
        unique_outcomes_tuples = {tuple(outcome) for outcome in all_successful_outcomes}
        return [list(outcome_tuple) for outcome_tuple in unique_outcomes_tuples]


    def _match_item_content(self, item_element, tokens):
        """ Matches content of an <item>. Can be text or sequence of other elements. """
        item_text = (item_element.text or "").strip().lower()
        
        # Children of <item> (e.g., <one-of>, <ruleref>, <tag> within <item>)
        children = list(item_element)

        if item_text and children: # Mixed content like <item> A <ruleref uri="#B"/> C </item>
            # This is complex. SRGS says text is tokenized. "A B C" is 3 tokens.
            # <item>A <ruleref uri="#foo"/> C</item>
            # Simplification: process text parts, then children.
            # For robust mixed content, one would tokenize the whole item content first.
            # For now, split text by children, treat as sequence.
            # This is a MAJOR simplification point. Proper mixed content tokenization is hard.
            # Let's assume text is either standalone or children are standalone for now.
            # If text exists, we match it. Then we try to match children with remaining tokens.
            # OR if no text, we match children.
            print(f"Warning: Mixed text and child elements in <item> '{item_element.get('id', 'anonymous')}' is handled simplistically.", file=sys.stderr)

        current_token_states = [tokens]

        # 1. Match leading text if present
        if item_text:
            next_token_states = []
            item_text_tokens = [t for t in item_text.split() if t] # Tokenize item text
            if not item_text_tokens: # Item text is empty or only whitespace
                 next_token_states.extend(current_token_states) # Consumes nothing
            else:
                for state in current_token_states:
                    if len(state) >= len(item_text_tokens) and \
                       state[:len(item_text_tokens)] == item_text_tokens:
                        next_token_states.append(state[len(item_text_tokens):])
            current_token_states = next_token_states
            if not current_token_states: # Text match failed
                return []

        # 2. Match children sequentially
        if children:
            return self._match_sequence(children, current_token_states)
        else: # No children, just text (or empty item)
            return current_token_states


    def _match_one_of_content(self, one_of_element, tokens):
        """ Matches content of <one-of>. Tries each <item> child. """
        all_possible_outcomes = []
        ns_map = {'gr': GRXML_NS}
        # <one-of> contains <item> children
        for item_child in one_of_element.findall("gr:item", ns_map):
            # Each item child is an alternative. Its repeat is implicitly 1 here.
            # The _match_element will handle the item's own repeat attr if it has one.
            # Here, we're concerned with one item *within* the one-of.
            results = self._match_element(item_child, tokens) # Match this specific item
            all_possible_outcomes.extend(results)
        
        unique_outcomes_tuples = {tuple(outcome) for outcome in all_possible_outcomes}
        return [list(outcome_tuple) for outcome_tuple in unique_outcomes_tuples]

    def _match_ruleref_content(self, ruleref_element, tokens):
        """ Matches a <ruleref>. Finds the rule and matches its content. """
        uri = ruleref_element.get("uri")
        special = ruleref_element.get("special") # NULL, VOID, GARBAGE

        if not uri and not special:
            raise GrxmlValidationError("Error: <ruleref> must have 'uri' or 'special' attribute.")

        if special: # Handle special rules
            if special == "NULL": # Matches empty string
                return [tokens]
            elif special == "VOID": # Never matches
                return []
            elif special == "GARBAGE": # Matches any speech up to next rule match or word
                # This is complex. For this simplified parser, GARBAGE might:
                # 1. Match one token (if greedy)
                # 2. Match nothing (if trying to allow subsequent matches)
                # Simplification: treat GARBAGE as optionally consuming one token.
                # Or better: not implemented fully. For now, let's say it can match nothing.
                # Or, it could be a "match any single token".
                print("Warning: <ruleref special='GARBAGE'> behavior is highly simplified (matches empty or one token).", file=sys.stderr)
                results = [tokens] # Match empty
                if tokens:
                    results.append(tokens[1:]) # Match one token
                return results

        if uri.startswith("#"):
            rule_id = uri[1:]
            if rule_id not in self.rules:
                raise GrxmlValidationError(f"Error: Undefined rule referenced: {rule_id}")
            
            target_rule_element = self.rules[rule_id]
            # The content of a rule is a sequence of its children
            return self._match_sequence(list(target_rule_element), tokens)
        else:
            raise GrxmlValidationError(f"Error: External URI or non-local ruleref '{uri}' not supported.")

    def _match_sequence(self, elements, initial_tokens_list_or_single):
        """
        Matches a sequence of elements against a list of current token states.
        `initial_tokens_list_or_single` can be a single token list [tok1, tok2]
        or a list of token lists [[tok1,tok2], [tokA,tokB]] representing alternative starting points.
        Returns a list of remaining token lists after the entire sequence matches.
        """
        if not isinstance(initial_tokens_list_or_single, list):
            raise TypeError("initial_tokens_list_or_single must be a list")

        if not elements: # Empty sequence matches, returns current states
             if initial_tokens_list_or_single and isinstance(initial_tokens_list_or_single[0], list):
                 return initial_tokens_list_or_single
             else: # It's a single token list
                 return [initial_tokens_list_or_single]


        # Ensure active_states is always a list of token lists
        if not initial_tokens_list_or_single or not isinstance(initial_tokens_list_or_single[0], list):
             active_states = [initial_tokens_list_or_single] # Wrap single token list
        else:
             active_states = initial_tokens_list_or_single


        for element in elements:
            next_active_states = []
            for current_tokens_state in active_states:
                # _match_element handles repeats for 'element'
                results_for_this_element = self._match_element(element, current_tokens_state)
                next_active_states.extend(results_for_this_element)
            
            active_states = next_active_states
            if not active_states: # Sequence broken, no way to match remaining elements
                return []
        
        return active_states


    def parse_string(self, input_string):
        """
        Tries to parse the input_string according to the loaded grammar.
        Returns True if the string is fully parsed by the root rule, False otherwise.
        """
        if not self.root_rule_id or self.root_rule_id not in self.rules:
            print(f"Error: Root rule '{self.root_rule_id}' is not defined or not found.", file=sys.stderr)
            return False

        tokens = [t for t in input_string.strip().lower().split() if t]
        
        root_rule_element = self.rules[self.root_rule_id]
        
        # The content of a rule is a sequence of its children elements
        # (e.g. <rule id="foo"><item>A</item><one-of>...</one-of></rule>)
        # We need to match this sequence.
        possible_outcomes = self._match_sequence(list(root_rule_element), tokens)

        for remaining_tokens in possible_outcomes:
            if not remaining_tokens:  # All tokens were consumed
                return True
        return False


def main():
    parser = argparse.ArgumentParser(description="ParseTool: Validates a .grxml file and optionally parses a string against it.")
    parser.add_argument("grxml_file", help="Path to the .grxml speech recognition grammar file.")
    parser.add_argument("-s", "--string", help="A quoted string to parse against the grammar.")

    args = parser.parse_args()

    try:
        grammar_parser = GrxmlParser(args.grxml_file) # This will validate on init
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
                 sys.exit(2) # Indicate parsing failure with a different exit code
        except GrxmlValidationError as e: # Errors during parsing logic (e.g. bad repeat)
            print(f"Error during string parsing attempt: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred during string parsing: {e}", file=sys.stderr)
            # import traceback
            # traceback.print_exc() # For debugging the parser logic
            sys.exit(1)
    else:
        # If -s is not provided, validation already happened and printed success.
        # We can just exit cleanly.
        print("GRXML file is valid (basic structural check). No string provided for parsing.", file=sys.stderr)
        pass

if __name__ == "__main__":
    main()