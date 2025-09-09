import json

with open(r"coffee_shop.json","r") as f:
    data = json.load(f)

def get_opening_hours(day):
    return data["shop_info"]["opening_hours"].get(day.lower(),"Sorry I don't have that day in my calendar")

def get_menu_items(item_name):
    for category, items in data["menu"].items():
        for item in items:
            if item_name.lower() in item["item"].lower():
                return f"{item['item']} costs {item['price']}. {item['description']}"
    return "Sorry, that item is not in the Menu"

def get_faq(question):
    for faq in data["faqs"]:
        if question.lower() in faq["question"].lower():
            return faq["answer"]
    return None
def chatbot():
    print("👋 Hello and Welcome to Adiel's coffe corner chatbot! (type 'quit' to exit")

    while True:
        user_input=input("You: ")
        if user_input.lower() == "quit":
            print("Bot: Goodbye! ☕")
            break
        faq_answer=get_faq(user_input)
        if faq_answer:
            print(f"Bot {faq_answer}")
            continue
        menu_answer=get_menu_items(user_input)
        if "Sorry" not in menu_answer:
            print(f"Bot: {menu_answer}")
            continue

        if "open" in user_input.lower():
            for day in data["shop_info"]["opening_hours"]:
                if day in user_input.lower():
                    if day in user_input.lower():
                        print(f"Bot: On {day.capitalize()}, we are open {get_opening_hours(day)}")
                        break
            else:
                print("Bot: We are open daily! Ask about a specific day for details.")
            continue
        print("Bot: I'm not sure about that. Ask me about the menu, hours or FAQs.")


if __name__ =="__main__":
    chatbot()

