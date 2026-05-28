def should_show_vehicle_cards(messages: list[dict], index: int, message: dict) -> bool:
    if message.get("role") != "assistant" or not message.get("vehicles"):
        return False
    if message.get("show_vehicle_cards") is False:
        return False
    if message.get("show_vehicle_cards") is True:
        return True
    last_idx = -1
    for idx, item in enumerate(messages):
        if item.get("role") == "assistant" and item.get("vehicles"):
            last_idx = idx
    return index == last_idx


def vehicle_card_title(message: dict) -> str:
    reserved = message.get("reserved") or (
        len(message.get("vehicles", [])) == 1
        and message.get("content", "").lower().startswith("done")
    )
    return "Reserved" if reserved else "Top picks"
