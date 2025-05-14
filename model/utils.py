from datetime import datetime

TYPE_DICTIONARY = {
    "bool": bool,
    "int": int,
    "float": float,
    "str": str,
    "date": datetime.date,
}


def create_message(message_text: str, role: str) -> dict:
    return {"role": role, "content": message_text}


def add_message_to_history(
    lst: list, msg: dict, static_head_num: int, max_tail_num: int
):
    lst.append(msg)
    tail_len = len(lst) - static_head_num
    if tail_len > max_tail_num:
        # Number of items to remove from just after the static head
        remove_count = tail_len - max_tail_num
        del lst[static_head_num : static_head_num + remove_count]
