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
from rasa_sdk.events import SlotSet
from ChessAI.main import ChessAI

ChessAI = ChessAI()


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

        return [SlotSet("move", ai_move)]


class ActionStartGame(Action):

    def name(self) -> Text:
        return "action_start_game"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        color = tracker.get_slot("color")
        if color == "schwarz":
            color = "black"
        else:
            color = "white"
        ChessAI.start_game(color, vs_ai=True)
        # Antworte dem Benutzer mit einer Bestätigungsnachricht
        dispatcher.utter_message(text="Das Spiel wurde gestartet!")

        # Setze den "game_started" Slot auf True, um zu verfolgen, dass das Spiel gestartet wurde
        return [SlotSet("game_started", True)]


class ActionQuitGame(Action):

    def name(self) -> Text:
        return "action_quit_game"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Hier code für das Beenden des Spiels einfügen
        ChessAI.quit_game()
        # Quit the game

        # Send a message to the user
        dispatcher.utter_message("Das Spiel wurde beendet!")

        return [SlotSet("game_started", False)]


class ActionRestartGame(Action):

    def name(self) -> Text:
        return "action_restart_game"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Hier code für das Neustarten des Spiels einfügen
        ChessAI.restart()
        # Restart the game

        # Send a message to the user
        dispatcher.utter_message("Das Spiel wurde neugestartet!")

        return [SlotSet("game_started", True)]
