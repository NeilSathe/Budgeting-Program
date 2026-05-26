
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

DATA_FOLDER = "Data"
USERS_FILE = os.path.join(DATA_FOLDER, "users.csv")

def register(username,password) :

    if os.path.exists(USERS_FILE) :
        df = pd.read_csv(USERS_FILE)
        if username in df["username"].values :
            print("Username Already Registered")
            return

    new_user = pd.DataFrame([[username,password]],columns=["username","password"])

    if os.path.exists(USERS_FILE):

        new_user.to_csv(USERS_FILE, mode="a", index=False, header=False)

    else :
        new_user.to_csv(USERS_FILE, index=False)

    print("Username Registered Successfully in Database")



def load_user_file(username) :

    return os.path.join(DATA_FOLDER, f"{username}_expenses.csv")

def load_df(username)  :

    file = load_user_file(username)
    return pd.read_csv(file)

def create_user_data(username) :

    file = load_user_file(username)

    if not os.path.exists(file):

        df = pd.DataFrame(columns=["month","index","cost","category","date"])
        df.to_csv(file, index=False)

def login(username,password)  :

    if not os.path.exists(USERS_FILE):

        print("No existing users in Database yet")
        return False


    df = pd.read_csv(USERS_FILE)

    for i,row in df.iterrows() :

        if row["username"] == username and row["password"] == password:
            print("Login Successful")
            create_user_data(username)
            return True


    print("Login Failed, Invalid Password or Username")
    return False

def add_expense(month, category, cost, date, username):

    file = load_user_file(username)

    if os.path.exists(file):

        df = pd.read_csv(file)

    else :

        df = pd.DataFrame(columns=["month","index","cost","category","date"])



    index = len(df) + 1

    new_expense = pd.DataFrame([{"month" : month,"index" : index,"cost" : cost,"category" : category,"date" : date}])
    df.loc[len(df)] = new_expense.iloc[0]
    df.to_csv(file, index=False)

def set_expense_limits(username,food,bills,clothes,online,fuel,groceries) :

    budgets = {"food" :food, "bills" : bills, "clothes" : clothes, "online" : online, "fuel" : fuel, "groceries" : groceries}

    total = sum(budgets.values())

    print(f"The total budget for this month is ${total}")

    check_cat_and_total(budgets,total,username)

    return budgets, total

def check_cat_and_total(budgets,total,username) :

    amounts = []

    df = load_df(username)

    spent = df.groupby("category")["cost"].sum()

    for category, budget in budgets.items():

        amount_spent = spent.get(category, 0)

        remaining = budget - amount_spent

        amounts.append(amount_spent)


        if amount_spent > budget:
            print(f"{category}: ${amount_spent} / ${budget} Over budget by ${remaining}")

        elif amount_spent >= (0.8 * budget):
            print(f" Category : {category}: Amount Spent : ${amount_spent} : Budget : ${budget} Warning : Expenditure is close to limit")

        else:
            print(f"Category : {category}: Amount Spent : ${amount_spent} : Budget :  ${budget} Expenditure is within budget")

    arr_amounts = np.array(amounts)

    amounts_sum = np.sum(arr_amounts)

    if amounts_sum > total:

        print(f"Warning : Expenditure is exceeding total budget of {total} by {amounts_sum}")

    elif amounts_sum >= (0.8 * total):

        print(f"Warning : Expenditure within but close to total budget of {total} by {total-amounts_sum}")

    else :

        print(f"Expenditure is comfortably within the budget of {total}. Remaining : {total-amounts_sum}")

def pie_expenditure(username) :

    file = load_df(username)

    cat_df = file.groupby("category")["cost"].sum()

    cat_df.plot(kind="pie", autopct="%1.1f%%")
    plt.title("Expenditure Distribution Across Categories")
    plt.xlabel("")
    plt.ylabel("")

    plt.show()

def delete_expense(index,username) :

    df = load_df(username)

    print(df.to_string(index=False))

    if index in df["index"].values :

        df = df[df["index"] != index]  # re number from 0 to start at 1

        df = df.reset_index(drop=True)
        df["index"] = df.index + 1

        df.to_csv(load_user_file(username), index=False)


        print("Expense successfully deleted")
        df = load_df(username)
        print(df.to_string(index=False))

    else :

        print("Expense not found")

def monthly_expenditure_prediction(username,month,category,date) :

    df = load_df(username)

    x = pd.get_dummies(df[["month", "category", "date"]]) #one hot encode category
    y = df["cost"]


    model = LinearRegression()
    model.fit(x,y)

    new_x = pd.DataFrame({
        "month": [month],
        "category": [category],
        "date": [date]
    })

    new_x = pd.get_dummies(new_x)
    new_x = new_x.reindex(columns=x.columns, fill_value=0)

    prediction = model.predict(new_x)[0]
    print(f"Predicted expense: ${prediction:.2f}")
    return prediction

def monthly_expenditure_graph(username) :

    df = load_df(username)

    data_group = df.groupby(["month","category"])["cost"].sum()

    data_group = data_group.unstack().fillna(0).sort_index()


    data_group.plot(kind="bar", stacked=True)

    plt.xlabel("Month")
    plt.ylabel("Expenditure")
    plt.title("Monthly Expenditure")

    plt.legend(title="Category")
    plt.tight_layout()

    plt.show()

def categorise_inputs(text) :

    df = pd.read_csv("general_category_class_training_data.csv")

    inputs_X = df["inputs"]

    labels = df["label"]

    vectorizer = CountVectorizer(lowercase=True)
    X = vectorizer.fit_transform(inputs_X)

    model = MultinomialNB()
    model.fit(X,labels)

    text = text.lower()
    new_x = vectorizer.transform([text])

    prediction = model.predict(new_x)[0]

    return prediction

#menu

print("Welcome to MyBudget!")

add = input("Do you want to register? y/n : ")

if add == "y" :

    username = input("Please enter your username: ")

    password = input("Please enter your password: ")

    register(username,password)



username = input("Please enter your username : ")

password = input("Please enter your password : ")

if not login(username,password)  :

    print("Login Failed, Program Closing")
    exit()

current_user = username


print("Menu : 1. Add Expense, 2. View Expenses (All), 3. Find total expenditure/set expenses, 4. Graph of Expenditure, \n5. Delete Expenditure, 6. Predict Expenditure, 7. Graph Monthly Expenditure, 8. Exit")
action = int(input("Enter Action : "))

while action not in [1,2,3,4,5,6,7,8] :

    print("Invalid Input. Enter a valid integer from 1-7 inclusive")
    action = int(input("Enter Action : "))

    if action in [1, 2, 3, 4, 5, 6, 7]:
        break

while action != 8 :

        if action == 1 :

                df = load_df(current_user)

                month = int(input("Enter Month : "))
                category = input("Enter Category : ")
                category = categorise_inputs(text=category)
                cost = float(input("Enter Cost : "))
                date = input("Enter Date : ")

                add_expense(month,category,cost,date,username)

                new_df = load_df(current_user)
                print(new_df.to_string(index=False))

                print("Menu : 1. Add Expense, 2. View Expenses (All), 3. Find total expenditure/set expenses, 4. Graph of Expenditure, \n5. Delete Expenditure, 6. Predict Expenditure, 7. Graph Monthly Expenditure, 8. Exit")
                action = int(input("Enter Action : "))

        elif action == 2 :

                df = load_df(current_user)
                print(df.to_string(index=False))

                print(
                    "Menu : 1. Add Expense, 2. View Expenses (All), 3. Find total expenditure/set expenses, 4. Graph of Expenditure, \n5. Delete Expenditure, 6. Predict Expenditure, 7. Graph Monthly Expenditure, 8. Exit")
                action = int(input("Enter Action : "))

        elif action == 3 :

                food = int(input("Enter Food : "))
                bills = int(input("Enter Bills : "))
                clothes = int(input("Enter Clothes : "))
                online = int(input("Enter Online : "))
                fuel = int(input("Enter Fuel : "))
                groceries = int(input("Enter Groceries : "))

                set_expense_limits(username,food,bills,clothes,online,fuel,groceries)

                print(
                    "Menu : 1. Add Expense, 2. View Expenses (All), 3. Find total expenditure/set expenses, 4. Graph of Expenditure, \n5. Delete Expenditure, 6. Predict Expenditure, 7. Graph Monthly Expenditure, 8. Exit")
                action = int(input("Enter Action : "))

        elif action == 4 :

                df = load_df(current_user)
                pie_expenditure(username)

                print(
                    "Menu : 1. Add Expense, 2. View Expenses (All), 3. Find total expenditure/set expenses, 4. Graph of Expenditure, \n5. Delete Expenditure, 6. Predict Expenditure, 7. Graph Monthly Expenditure, 8. Exit")
                action = int(input("Enter Action : "))

        elif action == 5 :

                df = load_df(current_user)
                print(df.to_string(index=False))
                index = int(input("Enter Index : "))
                delete_expense(index,current_user)

                print(
                    "Menu : 1. Add Expense, 2. View Expenses (All), 3. Find total expenditure/set expenses, 4. Graph of Expenditure, \n5. Delete Expenditure, 6. Predict Expenditure, 7. Graph Monthly Expenditure, 8. Exit")
                action = int(input("Enter Action : "))

        elif action == 6 :

                df = load_df(current_user)


                month = int(input("Enter Month : "))
                category = input("Enter Category : ")
                date = int(input("Enter Date : "))

                result = monthly_expenditure_prediction(current_user,month,category,date)
                print(f"Prediction: ${result:.2f}")

                print(
                    "Menu : 1. Add Expense, 2. View Expenses (All), 3. Find total expenditure/set expenses, 4. Graph of Expenditure, \n5. Delete Expenditure, 6. Predict Expenditure, 7. Graph Monthly Expenditure, 8. Exit")
                action = int(input("Enter Action : "))

        elif action == 7 :

                df = load_df(current_user)

                monthly_expenditure_graph(current_user)

                print(
                    "Menu : 1. Add Expense, 2. View Expenses (All), 3. Find total expenditure/set expenses, 4. Graph of Expenditure, \n5. Delete Expenditure, 6. Predict Expenditure, 7. Graph Monthly Expenditure, 8. Exit")
                action = int(input("Enter Action : "))


        elif action == 8:

            print("Thank you for using MyBudget")
            exit()




















































































