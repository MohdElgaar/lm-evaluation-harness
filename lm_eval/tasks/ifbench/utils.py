import dataclasses
from typing import Dict, Optional, Union

from lm_eval.tasks.ifbench import instructions_registry


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


def test_instruction_following_strict(inp, response):
    """Tests response to see if instructions are followed."""
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        kwargs = {k: v for k, v in inp.kwargs[index].items() if v is not None}
        arg_keys = instruction.get_instruction_args_keys()
        if arg_keys and "prompt" in arg_keys:
            kwargs["prompt"] = inp.prompt
        if arg_keys and "prompt_to_repeat" in arg_keys:
            kwargs["prompt_to_repeat"] = inp.prompt
        instruction.build_description(**kwargs)

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(inp, response):
    """Tests response for an upper bound for following instructions."""
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        kwargs = {k: v for k, v in inp.kwargs[index].items() if v is not None}
        arg_keys = instruction.get_instruction_args_keys()
        if arg_keys and "prompt" in arg_keys:
            kwargs["prompt"] = inp.prompt
        if arg_keys and "prompt_to_repeat" in arg_keys:
            kwargs["prompt_to_repeat"] = inp.prompt
        instruction.build_description(**kwargs)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def process_results(doc, results):
    inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"],
        kwargs=doc["kwargs"],
    )
    # results is [responses] from take_first_k (list of repeats) or [r0,r1,...] from multiple instances
    responses = results[0] if len(results) == 1 and isinstance(results[0], list) else results

    accs_strict = []
    accs_loose = []
    inst_strict = []
    inst_loose = []
    for response in responses:
        out_strict = test_instruction_following_strict(inp, response)
        out_loose = test_instruction_following_loose(inp, response)
        accs_strict.append(out_strict.follow_all_instructions)
        accs_loose.append(out_loose.follow_all_instructions)
        inst_strict.append(out_strict.follow_instruction_list)
        inst_loose.append(out_loose.follow_instruction_list)

    inst_strict_flat = [b for sublist in inst_strict for b in sublist]
    inst_loose_flat = [b for sublist in inst_loose for b in sublist]

    return {
        "prompt_level_strict_acc": sum(accs_strict) / len(accs_strict),
        "inst_level_strict_acc": inst_strict_flat,
        "prompt_level_loose_acc": sum(accs_loose) / len(accs_loose),
        "inst_level_loose_acc": inst_loose_flat,
    }


def agg_inst_level_acc(items):
    flat_items = [item for sublist in items for item in sublist]
    inst_level_acc = sum(flat_items) / len(flat_items)
    return inst_level_acc
