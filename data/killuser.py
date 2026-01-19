from database import delete_user_full

username = input("Welchen User willst du löschen? ")
sicher = input(f"Bist du sicher? Alle Portfolios von {username} werden gelöscht! (ja/nein): ")

if sicher.lower() == "ja":
    if delete_user_full(username):
        print(f"User '{username}' wurde gelöscht.")
    else:
        print("User nicht gefunden.")
else:
    print("Abgebrochen.")