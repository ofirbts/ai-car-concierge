from frontend.view_utils import should_show_vehicle_cards, vehicle_card_title


def test_should_show_vehicle_cards_honors_explicit_flag():
    msg = {"role": "assistant", "vehicles": [{"id": 1}], "show_vehicle_cards": False}
    assert should_show_vehicle_cards([msg], 0, msg) is False


def test_should_show_vehicle_cards_only_last_vehicle_message_without_flag():
    messages = [
        {"role": "assistant", "vehicles": [{"id": 1}]},
        {"role": "assistant", "content": "text only"},
        {"role": "assistant", "vehicles": [{"id": 2}]},
    ]
    assert should_show_vehicle_cards(messages, 0, messages[0]) is False
    assert should_show_vehicle_cards(messages, 2, messages[2]) is True


def test_vehicle_card_title_reserved_detection():
    assert vehicle_card_title({"reserved": True, "vehicles": [{"id": 1}], "content": "ok"}) == "Reserved"
    assert vehicle_card_title({"vehicles": [{"id": 1}], "content": "Done — reserved"}) == "Reserved"
    assert vehicle_card_title({"vehicles": [{"id": 1}], "content": "Top options"}) == "Top picks"
