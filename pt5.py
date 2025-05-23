#!/usr/bin/env python3

import argparse
import xml.etree.ElementTree as ET
import sys
import os
import re
import random

# --- Semantic Interpretation Imports and Setup ---
try:
    import js2py
    import json 
    js2py_available = True

    # Get js2py's internal representation of JavaScript 'undefined'
    _js_get_undefined_context = js2py.EvalJs()
    _js_get_undefined_context.execute('var _the_undefined_val = undefined;')
    JS_UNDEFINED_PROXY = _js_get_undefined_context._the_undefined_val

    class UndefinedMarker: 
        def __str__(self): return "internal_UNDEFINED_Marker"
        def __repr__(self): return "internal_UNDEFINED_Marker"
    UNDEFINED = UndefinedMarker()

except ImportError:
    js2py_available = False
    JS_UNDEFINED_PROXY = None 
    UNDEFINED = None 
    print("Warning: 'js2py' library not found. Semantic interpretation functionality will be disabled.", file=sys.stderr)
except Exception as e_js_setup: 
    js2py_available = False
    JS_UNDEFINED_PROXY = None 
    UNDEFINED = None 
    print(f"Warning: Error during js2py setup for UNDEFINED proxy: {e_js_setup}. Semantic interpretation may be unstable or disabled.", file=sys.stderr)
# --- End Semantic Interpretation Imports ---


GRXML_NS = "http://www.w3.org/2001/06/grammar"
XML_NS = "http://www.w3.org/XML/1998/namespace"

DEFAULT_GEN_CONFIG = {
    "max_additional_unbounded_repeats": 2, "max_absolute_repeats": 5,
    "max_recursion_depth": 25, "void_produces_failure": True,
    "garbage_placeholder": "[GARBAGE]", "max_single_sentence_attempts": 10
}

class GrxmlValidationError(Exception): pass
class GrxmlGenerationError(Exception): pass
class GrxmlSemanticError(Exception): pass

class SIContext:
    def __init__(self, initial_out=UNDEFINED):
        self.out = initial_out
        self.rules = {}
        self._latest_si = UNDEFINED

    def get_raw_vars_for_js_conversion(self):
        raw_vars = {'out': self.out}
        js_rules = self.rules.copy()
        js_rules['_latest'] = self._latest_si
        raw_vars['rules'] = js_rules
        return raw_vars

    def add_rule_si(self, rule_name, si_value):
        self._latest_si = si_value
        if rule_name in self.rules:
            current_val = self.rules[rule_name]
            if isinstance(current_val, list): current_val.append(si_value)
            else: self.rules[rule_name] = [current_val, si_value]
        else: self.rules[rule_name] = si_value
    
    def __repr__(self):
        return f"SIContext(out={self.out!r}, rules={self.rules!r}, _latest_si={self._latest_si!r})"

class GrxmlParser:
    _loaded_grammars_cache = {}

    def __init__(self, grxml_filepath, generation_config=None, is_external_load=False):
        if not os.path.isabs(grxml_filepath) and not is_external_load:
            self.filepath = os.path.abspath(grxml_filepath)
        else: self.filepath = os.path.normpath(grxml_filepath)
        self.tree, self.root_grammar_element, self.rules, self.root_rule_id = None, None, {}, None
        self.gen_config = DEFAULT_GEN_CONFIG.copy()
        if generation_config: self.gen_config.update(generation_config)
        self._validate_and_load()
        if self.filepath not in GrxmlParser._loaded_grammars_cache:
            GrxmlParser._loaded_grammars_cache[self.filepath] = self

    def _python_value_to_js_value_for_evaljs(self, value):
        if value is UNDEFINED: return JS_UNDEFINED_PROXY
        if isinstance(value, dict): return {k: self._python_value_to_js_value_for_evaljs(v) for k, v in value.items()}
        if isinstance(value, list): return [self._python_value_to_js_value_for_evaljs(i) for i in value]
        return value

    def _js_value_from_interpreter_to_python_value(self, js_value):
        if js_value is JS_UNDEFINED_PROXY:
            return UNDEFINED
        
        js_type_name = type(js_value).__name__

        if js_type_name == 'JsObjectWrapper': 
            try:
                temp_dict = {}
                is_dict_like = False
                if hasattr(js_value, ' लगभग') and isinstance(js_value. લગભગ, dict): # Check for internal dict structure
                    temp_dict = js_value. લગભગ
                    is_dict_like = True
                
                if not is_dict_like and hasattr(js_value, 'to_dict') and callable(js_value.to_dict): # Check for to_dict method
                    temp_dict = js_value.to_dict()
                    is_dict_like = True
                
                if not is_dict_like: # Try generic iteration for mapping-like objects
                    # Check if it's iterable and not a string-like wrapper to avoid iterating chars
                    if hasattr(js_value, '__iter__') and not (hasattr(js_value, 'is_string') and js_value.is_string()):
                        for key in js_value: 
                            py_key = key.to_python() if hasattr(key, 'to_python') else key
                            temp_dict[py_key] = js_value[key]
                        is_dict_like = True

                if is_dict_like:
                    return {self._js_value_from_interpreter_to_python_value(k_pyjs): 
                            self._js_value_from_interpreter_to_python_value(v_pyjs) 
                            for k_pyjs, v_pyjs in temp_dict.items()}
            except Exception: 
                pass # Fall through if iteration or specific attributes fail

            if hasattr(js_value, 'to_python') and callable(js_value.to_python):
                try:
                    python_val = js_value.to_python()
                    return self._recursively_map_js_proxies_in_py_object(python_val)
                except Exception:
                    return js_value # Fallback: return wrapper if to_python fails
            else:
                return js_value # Fallback: return wrapper if no other unwrapping method works

        if hasattr(js_value, 'to_python') and callable(js_value.to_python):
            try:
                python_val = js_value.to_python()
                return self._recursively_map_js_proxies_in_py_object(python_val)
            except Exception:
                return js_value 
        
        return js_value
    
    def _recursively_map_js_proxies_in_py_object(self, py_obj):
        if py_obj is JS_UNDEFINED_PROXY:
            return UNDEFINED
        
        if hasattr(py_obj, 'to_python') and callable(py_obj.to_python):
            return self._js_value_from_interpreter_to_python_value(py_obj)

        if isinstance(py_obj, dict):
            return {k: self._recursively_map_js_proxies_in_py_object(v) for k, v in py_obj.items()}
        if isinstance(py_obj, list):
            return [self._recursively_map_js_proxies_in_py_object(i) for i in py_obj]
        
        return py_obj

    def _execute_tag_script(self, script_content, si_context):
        if not js2py_available:
            return si_context
        raw_vars_for_js = si_context.get_raw_vars_for_js_conversion()
        js_vars_for_evaljs = self._python_value_to_js_value_for_evaljs(raw_vars_for_js)
        try:
            js_interpreter = js2py.EvalJs(js_vars_for_evaljs)
        except Exception as e_init:
            raise GrxmlSemanticError(f"Error initializing EvalJs for tag script in {self.filepath}:\nScript: '''{script_content}'''\nError: {e_init}\nSIContext: {si_context!r}\njs_vars: {js_vars_for_evaljs!r}") from e_init
        script_preamble = "var out = undefined;\n" if raw_vars_for_js['out'] is UNDEFINED else ""
        full_script = script_preamble + script_content
        try:
            js_interpreter.execute(full_script)
            if hasattr(js_interpreter, 'out'):
                si_context.out = self._js_value_from_interpreter_to_python_value(js_interpreter.out)
            else: si_context.out = UNDEFINED
        except Exception as e:
            raise GrxmlSemanticError(f"Error executing tag script in {self.filepath}:\nScript: '''{full_script}'''\nError: {e}\nSIContext: {si_context!r}\njs_vars: {js_vars_for_evaljs!r}") from e
        return si_context

    def _get_external_grammar(self, external_uri_path_str):
        current_grammar_dir = os.path.dirname(self.filepath)
        if os.path.isabs(external_uri_path_str): resolved_external_path = os.path.normpath(external_uri_path_str)
        else: resolved_external_path = os.path.normpath(os.path.join(current_grammar_dir, external_uri_path_str))
        if not os.path.exists(resolved_external_path): raise GrxmlValidationError(f"External grammar not found: {resolved_external_path}")
        if resolved_external_path in GrxmlParser._loaded_grammars_cache: return GrxmlParser._loaded_grammars_cache[resolved_external_path]
        return GrxmlParser(resolved_external_path, generation_config=self.gen_config, is_external_load=True)

    def _validate_and_load(self):
        if not os.path.exists(self.filepath): raise GrxmlValidationError(f"File not found: {self.filepath}")
        if not self.filepath.lower().endswith(".grxml"): raise GrxmlValidationError(f"Not .grxml: {self.filepath}")
        try: self.tree = ET.parse(self.filepath)
        except ET.ParseError as e: raise GrxmlValidationError(f"Invalid XML in {self.filepath}: {e}")
        self.root_grammar_element = self.tree.getroot()
        if self.root_grammar_element.tag != f"{{{GRXML_NS}}}grammar": raise GrxmlValidationError(f"Root not <grammar>: {self.root_grammar_element.tag}")
        self.root_rule_id = self.root_grammar_element.get("root")
        ns_map = {'gr': GRXML_NS}
        for rule_element in self.root_grammar_element.findall("gr:rule", ns_map):
            rule_id = rule_element.get("id")
            if not rule_id: raise GrxmlValidationError(f"<rule> lacks 'id' in {self.filepath}.")
            if rule_id in self.rules: raise GrxmlValidationError(f"Duplicate rule id '{rule_id}' in {self.filepath}.")
            self.rules[rule_id] = rule_element

    def _parse_repeat_attr(self, repeat_str):
        if not repeat_str: return 1, 1
        if repeat_str.isdigit(): val = int(repeat_str); return val, val
        if repeat_str == "optional": return 0, 1
        if repeat_str == "plus": return 1, float('inf')
        if repeat_str == "star": return 0, float('inf')
        if repeat_str.endswith("-"):
            try: min_r = int(repeat_str[:-1]); return min_r, float('inf')
            except ValueError: raise GrxmlValidationError(f"Invalid repeat format: {repeat_str}")
        match = re.match(r"(\d+)-(\d+)", repeat_str)
        if match:
            min_r, max_r = int(match.group(1)), int(match.group(2))
            if min_r > max_r: raise GrxmlValidationError(f"Invalid repeat range: {repeat_str}")
            return min_r, max_r
        if repeat_str == "0-1": return 0,1
        raise GrxmlValidationError(f"Unsupported repeat format: {repeat_str}")

    def _process_rule_for_match(self, rule_id_to_process, tokens, si_context_parent_for_rules_update, for_grammar_parser):
        if rule_id_to_process not in for_grammar_parser.rules:
             raise GrxmlValidationError(f"Rule '{rule_id_to_process}' not found in {for_grammar_parser.filepath}")
        rule_element = for_grammar_parser.rules[rule_id_to_process]
        rule_si_context = SIContext() 
        match_results_from_sequence = for_grammar_parser._match_sequence(
            list(rule_element), tokens, rule_si_context, si_context_parent_for_rules_update
        )
        final_results = []
        for rem_toks, _ in match_results_from_sequence:
            final_results.append((rem_toks, rule_si_context.out))
        return final_results

    def _match_element(self, element, tokens, current_si_context_for_element_out, parent_si_context_for_rules_update):
        tag_local_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag
        repeat_str = element.get('repeat', '1')
        min_reps, max_reps = self._parse_repeat_attr(repeat_str)
        possible_outcomes_after_repeats = []
        active_repeat_states = [(tokens, [])] 
        if min_reps == 0: possible_outcomes_after_repeats.append((tokens, UNDEFINED))
        effective_max_match_reps = max_reps
        if max_reps == float('inf'): effective_max_match_reps = len(tokens) + 1
        if min_reps > 0 and effective_max_match_reps < min_reps: effective_max_match_reps = min_reps
        if effective_max_match_reps > 20: effective_max_match_reps = 20 # Sanity cap

        for i_rep in range(1, int(effective_max_match_reps) + 1):
            if not active_repeat_states: break
            next_active_repeat_states = []
            for prev_tokens, accumulated_sis_list in active_repeat_states:
                expansion_si_context = SIContext()
                single_expansion_results = []
                if tag_local_name == "item": single_expansion_results = self._match_item_content(element, prev_tokens, expansion_si_context, parent_si_context_for_rules_update)
                elif tag_local_name == "one-of": single_expansion_results = self._match_one_of_content(element, prev_tokens, expansion_si_context, parent_si_context_for_rules_update)
                elif tag_local_name == "ruleref": single_expansion_results = self._match_ruleref_content(element, prev_tokens, expansion_si_context, parent_si_context_for_rules_update)
                elif tag_local_name == "token":
                    text_to_match = (element.text or "").strip().lower()
                    if prev_tokens and text_to_match and prev_tokens[0] == text_to_match: single_expansion_results.append((prev_tokens[1:], text_to_match))
                    elif not text_to_match: single_expansion_results.append((prev_tokens, UNDEFINED))
                for rem_toks, si_this_expansion in single_expansion_results:
                    new_accumulated_sis_list = accumulated_sis_list + [si_this_expansion]
                    next_active_repeat_states.append((rem_toks, new_accumulated_sis_list))
            active_repeat_states = next_active_repeat_states
            if i_rep >= min_reps:
                for rem_toks_after_rep, sis_list_for_rep_sequence in active_repeat_states:
                    final_si_for_reps = sis_list_for_rep_sequence
                    if len(sis_list_for_rep_sequence) == 1 and not isinstance(sis_list_for_rep_sequence[0], list): final_si_for_reps = sis_list_for_rep_sequence[0]
                    elif not sis_list_for_rep_sequence: final_si_for_reps = UNDEFINED
                    possible_outcomes_after_repeats.append((rem_toks_after_rep, final_si_for_reps))
        
        deduped_outcomes = []
        seen_outcomes = set()
        for r_tokens, r_si in possible_outcomes_after_repeats:
            si_key = tuple(r_si) if isinstance(r_si, list) else r_si
            si_key_for_set = "__UNDEFINED_MARKER__" if si_key is UNDEFINED else si_key # For set hashing
            outcome_key = (tuple(r_tokens), si_key_for_set)
            if outcome_key not in seen_outcomes:
                deduped_outcomes.append((r_tokens, r_si))
                seen_outcomes.add(outcome_key)
        return deduped_outcomes

    def _match_sequence(self, elements, tokens, si_context_for_sequence, parent_si_context_for_rules_update):
        if not elements: return [(tokens, si_context_for_sequence.out)]
        paths_being_explored = [(tokens, si_context_for_sequence)]

        for i, element in enumerate(elements):
            next_paths_being_explored = []
            tag_local_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag
            for path_tokens, path_si_context in paths_being_explored:
                if tag_local_name == "tag":
                    script_content = (element.text or "").strip()
                    if script_content:
                        try:
                            self._execute_tag_script(script_content, path_si_context)
                            next_paths_being_explored.append((path_tokens, path_si_context))
                        except GrxmlSemanticError as e: print(f"Semantic script error in sequence, path terminated: {e}", file=sys.stderr)
                    else: next_paths_being_explored.append((path_tokens, path_si_context))
                else:
                    results_for_element = self._match_element(element, path_tokens, path_si_context, parent_si_context_for_rules_update)
                    for rem_toks, element_si_value in results_for_element:
                        path_si_context._latest_si = element_si_value
                        next_paths_being_explored.append((rem_toks, path_si_context))
            paths_being_explored = next_paths_being_explored
            if not paths_being_explored: return []
        
        final_results = []
        for final_tokens, final_path_si_context_loop_var in paths_being_explored:
            final_results.append((final_tokens, final_path_si_context_loop_var.out))
        return final_results

    def _match_item_content(self, item_element, tokens, item_si_context, parent_si_context_for_rules_update):
        possible_item_outcomes = []
        tokens_after_leading_text = list(tokens) 
        leading_text_matched_and_consumed = False

        item_leading_text = (item_element.text or "").strip()
        if item_leading_text:
            item_leading_text_tokens = [t for t in item_leading_text.lower().split() if t]
            if len(tokens_after_leading_text) >= len(item_leading_text_tokens) and \
               tokens_after_leading_text[:len(item_leading_text_tokens)] == item_leading_text_tokens:
                tokens_after_leading_text = tokens_after_leading_text[len(item_leading_text_tokens):]
                leading_text_matched_and_consumed = True
                if not list(item_element): # No children, only this text
                    item_si_context.out = item_leading_text
            else:
                return [] # Leading text mismatch
        
        children = list(item_element)
        if children:
            results_from_children_sequence = self._match_sequence(
                children, tokens_after_leading_text, item_si_context, parent_si_context_for_rules_update
            )
            for rem_toks, _ in results_from_children_sequence:
                possible_item_outcomes.append((rem_toks, item_si_context.out))
        
        elif leading_text_matched_and_consumed : 
             possible_item_outcomes.append((tokens_after_leading_text, item_si_context.out))

        elif not item_leading_text and not children: 
            item_si_context.out = UNDEFINED
            possible_item_outcomes.append((tokens, item_si_context.out))
        
        return possible_item_outcomes

    def _match_one_of_content(self, one_of_element, tokens, one_of_si_context_ignored, parent_si_context_for_rules_update):
        all_possible_outcomes = []
        ns_map = {'gr': GRXML_NS}
        for item_child_element in one_of_element.findall("gr:item", ns_map):
            branch_item_si_context = SIContext()
            item_match_results = self._match_element(item_child_element, tokens, branch_item_si_context, parent_si_context_for_rules_update)
            for rem_toks, item_si_value in item_match_results:
                all_possible_outcomes.append((rem_toks, item_si_value))
        unique_outcomes_tuples, final_outcomes = set(), []
        for r_tokens, r_si in all_possible_outcomes:
            si_key = tuple(r_si) if isinstance(r_si, list) else r_si
            si_key_for_set = "__UNDEFINED_MARKER__" if si_key is UNDEFINED else si_key
            outcome_key = (tuple(r_tokens), si_key_for_set)
            if outcome_key not in unique_outcomes_tuples:
                final_outcomes.append((r_tokens, r_si)); unique_outcomes_tuples.add(outcome_key)
        return final_outcomes

    def _match_ruleref_content(self, ruleref_element, tokens, ruleref_si_context_ignored, parent_si_context_for_rules_update):
        uri = ruleref_element.get("uri", "N/A")
        special = ruleref_element.get("special")
        rule_name_for_parent_scope = ruleref_element.get("name")
        if not uri and not special: raise GrxmlValidationError(f"<ruleref> needs 'uri' or 'special'.")
        if special:
            su = special.upper()
            if su == "NULL": return [(tokens, UNDEFINED)]
            if su == "VOID": return []
            if su == "GARBAGE": return [(tokens[1:], tokens[0])] if tokens else [(tokens, UNDEFINED)]
            raise GrxmlValidationError(f"Unsupported special ruleref '{special}'.")
        target_parser, actual_rule_id = self, None
        if uri.startswith("#"): actual_rule_id = uri[1:]
        else:
            parts = uri.split('#', 1)
            ext_file, ext_frag = parts[0], (parts[1] if len(parts) > 1 else None)
            target_parser = self._get_external_grammar(ext_file)
            actual_rule_id = ext_frag if ext_frag else target_parser.root_rule_id
            if not actual_rule_id: raise GrxmlValidationError(f"No rule in {uri}, ext has no root.")
            if ext_frag and target_parser.rules[actual_rule_id].get("scope", "private") == "private":
                raise GrxmlValidationError(f"Ref to private external rule '{actual_rule_id}'.")
        if actual_rule_id not in target_parser.rules: raise GrxmlValidationError(f"Rule '{actual_rule_id}' not in {target_parser.filepath}")
        
        results_from_referenced_rule = target_parser._process_rule_for_match(
            actual_rule_id, tokens, parent_si_context_for_rules_update, target_parser
        )
        final_outcomes_for_ruleref = []
        for rem_toks, rule_out_si in results_from_referenced_rule:
            effective_name = rule_name_for_parent_scope if rule_name_for_parent_scope else actual_rule_id
            parent_si_context_for_rules_update.add_rule_si(effective_name, rule_out_si)
            final_outcomes_for_ruleref.append((rem_toks, rule_out_si))
        return final_outcomes_for_ruleref

    def parse_string(self, input_string, want_semantics=False):
        if not self.root_rule_id or self.root_rule_id not in self.rules:
            msg = f"Root rule '{self.root_rule_id}' not defined in {self.filepath}."
            if want_semantics: raise GrxmlSemanticError(msg)
            else: raise GrxmlValidationError(msg)
        tokens = [t for t in input_string.strip().lower().split() if t]
        overall_si_context = SIContext()
        match_results = self._process_rule_for_match(self.root_rule_id, tokens, overall_si_context, self)
        
        successful_parses_si = []
        for rem_toks, final_si in match_results:
            if not rem_toks:
                successful_parses_si.append(final_si)
        if not successful_parses_si:
            return None if want_semantics else False
        
        final_interpretation_object = successful_parses_si[0]
        
        if not want_semantics: return True
        if not js2py_available:
            print("Error: Semantic output requested, but js2py is not available.", file=sys.stderr)
            return None

        parser_self = self 
        def convert_for_json(obj):
            unwrapped_obj = parser_self._js_value_from_interpreter_to_python_value(obj)
            if unwrapped_obj is UNDEFINED or isinstance(unwrapped_obj, UndefinedMarker): 
                return None
            if isinstance(unwrapped_obj, dict): 
                return {k_dict: convert_for_json(v_dict) for k_dict, v_dict in unwrapped_obj.items()}
            if isinstance(unwrapped_obj, list): 
                return [convert_for_json(i_list) for i_list in unwrapped_obj]
            
            obj_type_name = type(unwrapped_obj).__name__
            if hasattr(unwrapped_obj, 'to_python') or obj_type_name.startswith('PyJs') or obj_type_name == 'JsObjectWrapper':
                # This path should ideally not be hit if _js_value_from_interpreter_to_python_value is exhaustive.
                return {"_serialization_error": f"Object of type {obj_type_name} is not JSON serializable after unwrapping. Value: {str(unwrapped_obj)[:100]}"}
            return unwrapped_obj
            
        instance_content_obj = convert_for_json(final_interpretation_object)
        try: 
            instance_content_str = json.dumps(instance_content_obj, ensure_ascii=False, indent=2)
        except TypeError as te: 
            raise GrxmlSemanticError(f"Failed to serialize final SI to JSON: {te}. Object after convert_for_json: {instance_content_obj!r}")

        xml_log = ET.Element("interpretation", grammar=self.filepath)
        ET.SubElement(xml_log, "instance").text = instance_content_str
        ET.SubElement(xml_log, "input", mode="voice").text = input_string
        return ET.tostring(xml_log, encoding="unicode")

    # --- Generation Methods (unchanged) ---
    def _determine_repetitions(self, min_reps, max_reps):
        eff_min = min_reps; eff_max = max_reps
        if max_reps == float('inf'): eff_max = min_reps + self.gen_config["max_additional_unbounded_repeats"]
        eff_max = min(eff_max, self.gen_config["max_absolute_repeats"])
        if eff_min > eff_max :
            if min_reps <= self.gen_config["max_absolute_repeats"]: eff_max = eff_min 
            else: eff_min = eff_max = self.gen_config["max_absolute_repeats"]
        if eff_min > eff_max: return eff_min 
        return random.randint(eff_min, int(eff_max))
    def _generate_random_expansion(self, element, depth):
        if depth > self.gen_config["max_recursion_depth"]: return ["[RECURSION_LIMIT]"] 
        tag_local_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag
        repeat_str = element.get('repeat', '1')
        try: min_reps, max_reps = self._parse_repeat_attr(repeat_str)
        except GrxmlValidationError: min_reps, max_reps = 1,1
        num_chosen_reps = self._determine_repetitions(min_reps, max_reps)
        if num_chosen_reps == 0: return []
        all_generated_words = []
        for _ in range(num_chosen_reps):
            words_from_one_expansion = self._expand_element_content_once(element, depth + 1)
            if words_from_one_expansion is None: return None 
            all_generated_words.extend(words_from_one_expansion)
        return all_generated_words
    def _expand_element_content_once(self, element_def, depth):
        tag_local_name = element_def.tag.split('}')[-1] if '}' in element_def.tag else element_def.tag
        generated_words = []
        ns_map = {'gr': GRXML_NS}
        if tag_local_name == "item":
            item_text = (element_def.text or "").strip()
            if item_text: generated_words.extend(item_text.split())
            for child_element in list(element_def):
                child_expansion = self._generate_random_expansion(child_element, depth)
                if child_expansion is None: return None
                generated_words.extend(child_expansion)
        elif tag_local_name == "one-of":
            item_choices = element_def.findall("gr:item", ns_map)
            if not item_choices: return None 
            chosen_item = random.choice(item_choices)
            item_expansion = self._generate_random_expansion(chosen_item, depth)
            if item_expansion is None: return None
            generated_words.extend(item_expansion)
        elif tag_local_name == "ruleref":
            uri = element_def.get("uri"); special = element_def.get("special")
            if not uri and not special: raise GrxmlGenerationError(f"<ruleref> needs 'uri' or 'special'.")
            if special:
                su = special.upper()
                if su == "NULL": pass
                elif su == "VOID": return None
                elif su == "GARBAGE": generated_words.append(self.gen_config["garbage_placeholder"])
                else: raise GrxmlGenerationError(f"Unsupported special ruleref '{special}'.")
            elif uri.startswith("#"):
                rule_id = uri[1:]
                if rule_id not in self.rules: raise GrxmlGenerationError(f"Local rule '{rule_id}' not in {self.filepath}.")
                target_rule_element = self.rules[rule_id]
                for child_in_rule in list(target_rule_element):
                    expansion = self._generate_random_expansion(child_in_rule, depth)
                    if expansion is None: return None
                    generated_words.extend(expansion)
            else: 
                parts=uri.split('#',1);ext_file,ext_frag=parts[0],(parts[1] if len(parts)>1 else None)
                ext_parser=self._get_external_grammar(ext_file)
                target_id=ext_frag if ext_frag else ext_parser.root_rule_id
                if not target_id: raise GrxmlGenerationError(f"No rule in {uri}, ext has no root.")
                if target_id not in ext_parser.rules: raise GrxmlGenerationError(f"Rule '{target_id}' not in {ext_parser.filepath}.")
                target_rule_ext=ext_parser.rules[target_id]
                if ext_frag and target_rule_ext.get("scope","private")=="private": raise GrxmlGenerationError(f"Gen from private external rule '{target_id}'.")
                for child_in_ext_rule in list(target_rule_ext):
                    expansion = ext_parser._generate_random_expansion(child_in_ext_rule, depth)
                    if expansion is None: return None
                    generated_words.extend(expansion)
        elif tag_local_name == "token":
            token_text = (element_def.text or "").strip();
            if token_text: generated_words.append(token_text)
        elif tag_local_name == "tag": pass
        return generated_words
    def generate_sentences(self, num_sentences):
        if not self.root_rule_id or self.root_rule_id not in self.rules: raise GrxmlGenerationError(f"Root rule '{self.root_rule_id}' not defined.")
        root_rule_element = self.rules[self.root_rule_id]; generated_count = 0
        for i in range(num_sentences):
            sentence_words = None
            for _ in range(self.gen_config["max_single_sentence_attempts"]):
                current_sentence_words, gen_ok = [], True
                for child_of_root in list(root_rule_element):
                    expansion = self._generate_random_expansion(child_of_root, 0)
                    if expansion is None: gen_ok = False; break
                    current_sentence_words.extend(expansion)
                if gen_ok: sentence_words = current_sentence_words; break 
            if sentence_words is not None: print(" ".join(s for s in sentence_words if s)); generated_count += 1
            else: print(f"Warn: Could not gen sentence {i+1}.", file=sys.stderr)
        if generated_count < num_sentences: print(f"Info: Gen {generated_count}/{num_sentences}.", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="ParseTool for GRXML with optional Semantic Interpretation.")
    parser.add_argument("grxml_file", help="Path to the main .grxml file.")
    parser.add_argument("-s", "--string", help="A quoted string to parse.")
    parser.add_argument("-g", "--generate", type=int, metavar="N", help="Generate N sentences.")
    parser.add_argument("--semantic-output", action="store_true", help="Enable Semantic Interpretation and output XML log if parsing with -s.")
    args = parser.parse_args()
    initial_grammar_path = args.grxml_file

    if args.string is None and args.generate is None:
        try: GrxmlParser(initial_grammar_path); print(f"GRXML {initial_grammar_path} basic validation OK.", file=sys.stderr)
        except Exception as e: print(f"Error: {e}", file=sys.stderr); sys.exit(1)
        sys.exit(0)

    grammar_parser = None
    try: grammar_parser = GrxmlParser(initial_grammar_path)
    except Exception as e: print(f"Error loading grammar {initial_grammar_path}: {e}", file=sys.stderr); sys.exit(1)

    if args.string is not None:
        if args.semantic_output and not js2py_available:
            print("Error: --semantic-output, but 'js2py' not available.", file=sys.stderr); sys.exit(1)
        try:
            result = grammar_parser.parse_string(args.string, want_semantics=args.semantic_output)
            if args.semantic_output:
                if result: print(result)
                else: print("False"); sys.exit(2)
            else:
                print(result)
                if not result: sys.exit(2)
        except GrxmlSemanticError as e: print(f"Semantic Interpretation Error: {e}", file=sys.stderr); sys.exit(1)
        except Exception as e: print(f"Error during string parsing: {e}", file=sys.stderr); sys.exit(1)
            
    if args.generate is not None:
        if args.generate <= 0: print("Error: -g N must be positive.", file=sys.stderr); sys.exit(1)
        try: grammar_parser.generate_sentences(args.generate)
        except Exception as e: print(f"Error during sentence generation: {e}", file=sys.stderr); sys.exit(1)

if __name__ == "__main__":
    main()