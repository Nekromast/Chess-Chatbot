# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions
from typing import Any, Dict, List, Text

# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from ChessAI.main import ChessAI

class ActionMakeMove(Action):

    def name(self) -> Text:
        return "action_make_move"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Get the user's move from the tracker
        move = tracker.get_slot("move")

        # Send the move to the chess AI and receive the AI's move
        ai_move = ChessAI.make_move(move)

        # Send the AI's move back to the user via the chatbot
        dispatcher.utter_message("My move is: {}".format(ai_move))

        return []
