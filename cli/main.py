"""Zentraler, terminalbasierter Einstiegspunkt ohne vorzeitiges Modellladen."""

from cli.common import menu, model_menu, say


def main():
    while True:
        try:
            choice = menu("CHAPPiE Activation Steering", [
                ("1", "Chat"), ("2", "Forschung"), ("3", "Modell wechseln"), ("4", "Beenden"),
            ])
            if choice == "1":
                from cli.chat import chat_menu
                chat_menu()
            elif choice == "2":
                from cli.research import research_menu
                research_menu()
            elif choice == "3":
                model_menu()
            elif choice in ("4", "q"):
                return 0
            else:
                say("Bitte 1, 2, 3 oder 4 wählen.")
        except KeyboardInterrupt:
            say("\nZurück zum Hauptmenü.")
        except EOFError:
            say("\nCHAPPiE beendet.")
            return 0
        except (ValueError, RuntimeError, OSError) as error:
            say(f"Aktion nicht möglich: {error}")


if __name__ == "__main__":
    raise SystemExit(main())
