import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# url = "https://api.coingecko.com/api/v3/coins/list"
#
# response = requests.get(url)
# data = response.json()
#
# # Extract the `{}` values from the response
# coin_symbols = [coin['id'] for coin in data]
#
# # Print each value on a different row
# for symbol in coin_symbols:
#     print(symbol)

def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period + 1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100.0 - 100.0 / (1.0 + rs)

    divergences = []

    for i in range(period, len(prices)):
        delta = deltas[i - 1]  # Price change since yesterday
        if delta > 0:
            upval = delta
            downval = 0.0
        else:
            upval = 0.0
            downval = -delta

        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)

        # Check for divergences
        current_rsi = rsi[i]
        last_rsi_high = np.max(rsi[i - 60:i]) if i >= 60 else np.max(rsi[:i])
        last_rsi_low = np.min(rsi[i - 60:i]) if i >= 60 else np.min(rsi[:i])

        # Check for divergences
        current_rsi = rsi[i]
        if current_rsi > last_rsi_high and last_rsi_high < 70:
            # Regular bearish divergence
            divergence = (i, "BEAR")
            divergences.append(divergence)
        elif current_rsi < last_rsi_low and last_rsi_low > 30:
            # Regular bullish divergence
            divergence = (i, "BULL")
            divergences.append(divergence)
        elif current_rsi < last_rsi_high and last_rsi_high > 70:
            # Hidden bearish divergence
            divergence = (i, "H-BEAR")
            divergences.append(divergence)
        elif current_rsi > last_rsi_low and last_rsi_low < 30:
            # Hidden bullish divergence
            divergence = (i, "H-BULL")
            divergences.append(divergence)

        # Update last RSI values
        last_rsi_high = max(current_rsi, last_rsi_high)
        last_rsi_low = min(current_rsi, last_rsi_low)

    return rsi, divergences





def calculate_stochastic_rsi(rsi, k_period=3, d_period=3):
    rsi_range = rsi.max() - rsi.min()
    stoch_rsi = (rsi - rsi.min()) / rsi_range

    stoch_rsi = pd.Series(stoch_rsi)  # Convert to pandas Series

    stoch_rsi_k = stoch_rsi.rolling(k_period).mean()
    stoch_rsi_d = stoch_rsi_k.rolling(d_period).mean()

    return stoch_rsi_k, stoch_rsi_d

def get_charts():
    get_current_stats()



    api_url = "https://api.coingecko.com/api/v3/coins/{}/market_chart"

    figs = []
    days = input("input the number of DAYS to analyze: ")
    n_days = eval(days)
    for symbol in crypto_symbols:
        # Step 2: Retrieve historical price data
        url = api_url.format(symbol)
        params = {
            "vs_currency": "usd",
            "days": n_days
        }
        response = requests.get(url, params=params)
        data = response.json()
        prices = np.array(data["prices"])[:, 1]  # Extract closing prices

        # Step 3: Calculate RSI
        rsi = calculate_rsi(prices)

        # Step 4: Calculate Stochastic RSI
        stoch_rsi_k, stoch_rsi_d = calculate_stochastic_rsi(rsi)

        # Step 5: Plot the results
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"{symbol.upper()} - Price, RSI, and Stochastic RSI")

        # Plot Price
        ax1.semilogy(prices, label="Price")
        ax1.set_ylabel("Price (USD)")

        ax1.grid(True)

        # Plot RSI
        ax2.plot(rsi, label="RSI")
        ax2.plot([30] * len(rsi), "--", color="red", label="RSI 30")
        ax2.plot([70] * len(rsi), "--", color="green", label="RSI 70")
        ax2.set_ylabel("RSI Value")
        ax2.grid(True)

        # Plot Stochastic RSI
        ax3.plot(stoch_rsi_k, label="%K")
        ax3.plot(stoch_rsi_d, label="%D")
        ax3.plot([0.2] * len(stoch_rsi_k), "--", color="red", label="Stochastic RSI 20")
        ax3.plot([0.8] * len(stoch_rsi_k), "--", color="green", label="Stochastic RSI 80")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Stochastic RSI Value")
        ax3.grid(True)

        figs.append(fig)  # Append the figure to the list

    # Show all figures at once
    for fig in figs:
        fig.show()
    plt.show()
    menu()


def get_current_stats():
    if not os.path.exists("Genesis_data.csv"):
        make_coins()
    else:
        data = pd.read_csv("Genesis_data.csv")
        global crypto_symbols
        crypto_symbols = data["name_lower"].tolist()
        print(crypto_symbols)


def bot_signal():
    get_current_stats()
    print("")
    print(crypto_symbols)
    print("")


    url = "https://api.coingecko.com/api/v3/coins/{}/market_chart"


    coin_dfs = {}  # Dictionary to store DataFrames for each coin
    for symbol in crypto_symbols:
        symbol_url = url.format(symbol)
        params = {
            "vs_currency": "usd",
            "days": (365)
        }
        response = requests.get(symbol_url, params=params)
        print("RESPONSE----------------------------- RESPONSE --------------------------------------- RESPONSE")
        print(response)
        print("RESPONSE----------------------------- RESPONSE --------------------------------------- RESPONSE", symbol)

        data = response.json()
        print("DATA ----------------------------- DATA --------------------------------------- DATA")
        print(data)
        print("DATA ----------------------------- DATA ----------------------------------------DATA", symbol)

        try:
            prices = np.array(data["prices"])
        except KeyError:
            print("KeyError: 'prices'. Waiting for 1 minute.")
            for i in range(60):
                print(f"Sleeping for {60 - i} seconds...")
                time.sleep(1)  # Sleep for 1 second
            continue

        dates = pd.to_datetime(prices[:, 0], unit='ms').floor("D")  # Floor the dates to day level
        prices = prices[:, 1]

        # Create a DataFrame for the symbol
        df = pd.DataFrame({"Date": dates, "Price": prices})
        df["Symbol"] = symbol  # Add a column for the symbol


        # Calculate RSI
        rsi, divergences = calculate_rsi(prices)  # Get RSI and divergences
        df = pd.DataFrame({"Date": dates, "Price": prices, "RSI": rsi})  # Create DataFrame
        df["Symbol"] = symbol  # Add a column for the symbol

        coin_dfs[symbol] = df

        # Calculate Stochastic RSI
        stoch_rsi_k, stoch_rsi_d = calculate_stochastic_rsi(rsi)
        df["StochRSI_K"] = stoch_rsi_k
        df["StochRSI_D"] = stoch_rsi_d

        # Calculate the median price for each day
        df = df.groupby("Date").agg(
            {"Price": "median", "RSI": "last", "StochRSI_K": "last", "StochRSI_D": "last"}).reset_index()

        coin_dfs[symbol] = df  # Store the DataFrame for the coin

    # Access the DataFrames for each coin separately
    for symbol, df in coin_dfs.items():
        print(f"DataFrame for {symbol}:")
        print(df)
        print()


    for symbol, df in coin_dfs.items():
        rsi, divergences = calculate_rsi(df["Price"])
        print(f"Divergences for {symbol}:")
        for divergence in divergences:
            index, divergence_type = divergence
            print(f"At index {index}: {divergence_type} divergence")
        print()
        # Create a new figure and subplots for each coin
        fig, axes = plt.subplots(4, 1, figsize=(15, 8))
        fig.suptitle(f"Price, RSI, and Stochastic RSI for {symbol.upper()}")

        # Plot 1: Price
        axes[0].plot(df["Date"], df["Price"])
        axes[0].set_ylabel("Price")
        axes[0].set_xlabel("Date")
        axes[0].set_yscale("log")
        axes[0].grid(True)

        # Plot 2: RSI
        axes[1].plot(df["Date"], df["RSI"], color="green")
        axes[1].set_ylabel("RSI")
        axes[1].set_xlabel("Date")
        axes[1].set_ylim(0, 100)  # Set the y-axis limit
        axes[1].axhline(y=70, color="red", linestyle="--", linewidth=1)  # Add line at 70
        axes[1].axhline(y=50, color="orange", linestyle="--", linewidth=1)  # Add line at 50
        axes[1].axhline(y=30, color="red", linestyle="--", linewidth=1)  # Add line at 30
        axes[1].grid(True)  # Show grid

        # Print divergences on RSI plot
        for divergence in divergences:
            index, divergence_type = divergence
            rsi_value = df.loc[index, "RSI"]
            date = df.loc[index, "Date"]

            if divergence_type == "H-BULL":
                color = "green"
            elif divergence_type == "BULL":
                color = "green"
            elif divergence_type == "BEAR":
                color = "red"
            elif divergence_type == "H-BEAR":
                color = "red"
            else:
                color = "black"

            axes[1].axvline(x=date, color=color, linestyle="--", linewidth=1)

        # Plot 3: Stochastic RSI
        axes[2].plot(df["Date"], df["StochRSI_K"], color="blue", label="StochRSI K")
        axes[2].plot(df["Date"], df["StochRSI_D"], color="orange", label="StochRSI D")
        axes[2].set_ylabel("Stochastic RSI")
        axes[2].set_xlabel("Date")
        axes[2].set_ylim(0, 1)  # Set the y-axis limit
        axes[2].axhline(y=0.8, color="red", linestyle="--", linewidth=1)  # Add line at 0.8
        axes[2].axhline(y=0.5, color="orange", linestyle="--", linewidth=1)  # Add line at 0.5
        axes[2].axhline(y=0.2, color="red", linestyle="--", linewidth=1)  # Add line at 0.2

        # Highlight line and add vertical lines when value is below 0.25 (green) or above 0.7 (red) with corresponding momentum conditions
        stoch_rsi_k = df["StochRSI_K"]
        momentum_up = stoch_rsi_k > stoch_rsi_k.shift(1)
        momentum_down = stoch_rsi_k < stoch_rsi_k.shift(1)
        axes[2].plot(df["Date"], stoch_rsi_k, color="blue", linewidth=1)  # Plot the original line
        axes[2].plot(df["Date"], stoch_rsi_k.where((stoch_rsi_k < 0.25) & momentum_up), color="green",
                     linewidth=1)  # Highlight line when value is below 0.25 and momentum is up


        axes[2].plot(df["Date"], stoch_rsi_k.where((stoch_rsi_k > 0.7) & momentum_down), color="red",
                     linewidth=1)  # Highlight line when value is above 0.7 and momentum is down

        highlight_dates_green = df.loc[(stoch_rsi_k < 0.25) & momentum_up, "Date"]  # Get the dates for green lines
        highlight_dates_red = df.loc[(stoch_rsi_k > 0.7) & momentum_down, "Date"]  # Get the dates for red lines
        for date in highlight_dates_green:
            axes[2].axvline(x=date, color="green", linestyle="--", linewidth=1)  # Add green vertical line
        for date in highlight_dates_red:
            axes[2].axvline(x=date, color="red", linestyle="--", linewidth=1)  # Add red vertical line

        axes[2].grid(True)  # Show grid

        for i in range(len(df)):
            if df["RSI"].iloc[i] <= 33 and df["StochRSI_K"].iloc[i] <= 0.25:
                axes[0].axvline(x=df["Date"].iloc[i], color="green", linestyle="-", linewidth=2)

            if df["RSI"].iloc[i] > 30 and df["RSI"].iloc[i] < 40 and df["StochRSI_K"].iloc[i] < 0.25:
                axes[0].axvline(x=df["Date"].iloc[i], color="green", linestyle="--", linewidth=1)

            if df["RSI"].iloc[i] <= 30 and df["StochRSI_K"].iloc[i] <= 0.36:
                axes[0].axvline(x=df["Date"].iloc[i], color="green", linestyle="-", linewidth=2)



            if df["RSI"].iloc[i] >= 70 and df["StochRSI_K"].iloc[i] >= 0.8:
                axes[0].axvline(x=df["Date"].iloc[i], color="red", linestyle="-", linewidth=2)

            if df["RSI"].iloc[i] >= 65 and df["StochRSI_K"].iloc[i] <= 0.75:
                axes[0].axvline(x=df["Date"].iloc[i], color="red", linestyle="--", linewidth=1)



            if df["RSI"].iloc[i] < 70 and df["RSI"].iloc[i] > 50 and df["StochRSI_K"].iloc[i] < 0.7 and df["StochRSI_K"].iloc[i] > 0.55:
                axes[0].axvline(x=df["Date"].iloc[i], color="orange", linestyle="--", linewidth=1)

            if df["RSI"].iloc[i] < 55 and df["RSI"].iloc[i] > 45 and df["StochRSI_K"].iloc[i] < 0.55 and df["StochRSI_K"].iloc[i] > 0.45:
                axes[0].axvline(x=df["Date"].iloc[i], color="yellow", linestyle="-", linewidth=2)







        # Plot 4: Vertical Lines for Green Signals
        axes[3].set_ylabel("Custom Conditions")
        axes[3].set_xlabel("Date")
        axes[3].grid(True)
        axes[3].set_yscale("log")
        axes[3].plot(df["Date"], df["Price"], color='black', linestyle='-', linewidth=0.5)


        stoch_rsi_k = df["StochRSI_K"]
        momentum_up = stoch_rsi_k > stoch_rsi_k.shift(1)
        momentum_down = stoch_rsi_k < stoch_rsi_k.shift(1)

        # Checking if lines exist in axes[0], axes[1], and axes[2]
        for i in range(len(df)):
            if df["RSI"].iloc[i] <= 33 and df["StochRSI_K"].iloc[i] <= 0.25 and \
                df["StochRSI_K"].iloc[i] < 0.25 and momentum_up[i] and \
                    "H-BULL" in [div[1] for div in divergences]:
                        axes[3].axvline(x=df["Date"].iloc[i], color='green', linestyle='--', linewidth=1.0)

        for i in range(len(df)):
            if df["RSI"].iloc[i] <= 33 and df["StochRSI_K"].iloc[i] <= 0.25 and \
                df["StochRSI_K"].iloc[i] < 0.25 and momentum_up[i] and \
                    "BULL" in [div[1] for div in divergences]:
                        axes[3].axvline(x=df["Date"].iloc[i], color='green', linestyle='-', linewidth=1.0)


        for i in range(len(df)):
            if df["RSI"].iloc[i] > 30 and df["RSI"].iloc[i] < 40 and df["StochRSI_K"].iloc[i] <= 0.25 and \
                df["StochRSI_K"].iloc[i] < 0.25 and momentum_up[i] and \
                    "H-BULL" in [div[1] for div in divergences]:
                        axes[3].axvline(x=df["Date"].iloc[i], color='green', linestyle='--', linewidth=1.0)

        for i in range(len(df)):
            if df["RSI"].iloc[i] > 30 and df["RSI"].iloc[i] < 40 and df["StochRSI_K"].iloc[i] <= 0.25 and \
                df["StochRSI_K"].iloc[i] < 0.25 and momentum_up[i] and \
                    "BULL" in [div[1] for div in divergences]:
                        axes[3].axvline(x=df["Date"].iloc[i], color='green', linestyle='-', linewidth=1.0)

        for i in range(len(df)):
            if df["RSI"].iloc[i] <= 30 and df["StochRSI_K"].iloc[i] <= 0.36 and \
                df["StochRSI_K"].iloc[i] < 0.25 and momentum_up[i] and \
                    "H-BULL" in [div[1] for div in divergences]:
                        axes[3].axvline(x=df["Date"].iloc[i], color='green', linestyle='--', linewidth=1.0)

        for i in range(len(df)):
            if df["RSI"].iloc[i] <= 30 and df["StochRSI_K"].iloc[i] <= 0.36 and \
                df["StochRSI_K"].iloc[i] < 0.25 and momentum_up[i] and \
                    "BULL" in [div[1] for div in divergences]:
                        axes[3].axvline(x=df["Date"].iloc[i], color='green', linestyle='-', linewidth=1.0)



        for i in range(len(df)):
            if df["RSI"].iloc[i] >= 70 and df["StochRSI_K"].iloc[i] >= 0.8 and \
                df["StochRSI_K"].iloc[i] > 0.7 and momentum_down[i] and \
                    "BEAR" in [div[1] for div in divergences]:
                        axes[3].axvline(x=df["Date"].iloc[i], color='red', linestyle='-', linewidth=1.0)

        for i in range(len(df)):
            if df["RSI"].iloc[i] >= 70 and df["StochRSI_K"].iloc[i] >= 0.8 and \
                df["StochRSI_K"].iloc[i] > 0.7 and momentum_down[i] and \
                    "H-BEAR" in [div[1] for div in divergences]:
                        axes[3].axvline(x=df["Date"].iloc[i], color='red', linestyle='--', linewidth=1.0)

        for i in range(len(df)):
            if df["RSI"].iloc[i] >= 65 and df["StochRSI_K"].iloc[i] <= 0.75 and \
                df["StochRSI_K"].iloc[i] > 0.7 and momentum_down[i] and \
                    "BEAR" in [div[1] for div in divergences]:
                        axes[3].axvline(x=df["Date"].iloc[i], color='red', linestyle='--', linewidth=1.0)

        for i in range(len(df)):
            if df["RSI"].iloc[i] >= 65 and df["StochRSI_K"].iloc[i] <= 0.75 and \
                df["StochRSI_K"].iloc[i] > 0.7 and momentum_down[i] and \
                    "H-BEAR" in [div[1] for div in divergences]:
                        axes[3].axvline(x=df["Date"].iloc[i], color='red', linestyle='-', linewidth=1.0)

        for i in range(len(df)):
            if df["RSI"].iloc[i] < 70 and df["RSI"].iloc[i] > 50 and df["StochRSI_K"].iloc[i] < 0.7 and df["StochRSI_K"].iloc[i] > 0.55 and \
                df["StochRSI_K"].iloc[i] > 0.7 and momentum_down[i] and \
                    "BEAR" in [div[1] for div in divergences]:
                        axes[3].axvline(x=df["Date"].iloc[i], color='red', linestyle='-', linewidth=1.0)
        for i in range(len(df)):
            if df["RSI"].iloc[i] < 70 and df["RSI"].iloc[i] > 50 and df["StochRSI_K"].iloc[i] < 0.7 and df["StochRSI_K"].iloc[i] > 0.55 and \
                df["StochRSI_K"].iloc[i] > 0.7 and momentum_down[i] and \
                    "H-BEAR" in [div[1] for div in divergences]:
                        axes[3].axvline(x=df["Date"].iloc[i], color='red', linestyle='--', linewidth=1.0)


        #
        # stoch_rsi_k = df["StochRSI_K"]
        # momentum_up = stoch_rsi_k > stoch_rsi_k.shift(1)
        # momentum_down = stoch_rsi_k < stoch_rsi_k.shift(1)
        #
        # axes[2].plot(df["Date"], stoch_rsi_k, color="blue", linewidth=1)  # Plot the original line
        #
        # for i in range(len(df)):
        #     if (df["RSI"].iloc[i] <= 33 and df["StochRSI_K"].iloc[i] <= 0.25):
        #             if (momentum_up.iloc[i] and divergence_type == "H-BULL"):
        #                 axes[3].axvline(x=df["Date"].iloc[i], color='green', linestyle='--', linewidth=1.0)
        #
        #
        #     elif (df["RSI"].iloc[i] <= 33 and df["StochRSI_K"].iloc[i] <= 0.25):
        #             if (momentum_up.iloc[i] and divergence_type == "BULL"):
        #                 axes[3].axvline(x=df["Date"].iloc[i], color='green', linestyle='-', linewidth=1.0)
        #
        #
        #     elif (df["RSI"].iloc[i] >30 and df["RSI"].iloc[i] <40 and df["StochRSI_K"].iloc[i] < 0.25):
        #             if (momentum_up.iloc[i] and divergence_type == "H-BULL"):
        #                 axes[3].axvline(x=df["Date"].iloc[i], color='green', linestyle='--', linewidth=1.0)
        #
        #     elif (df["RSI"].iloc[i] >30 and df["RSI"].iloc[i] <40 and df["StochRSI_K"].iloc[i] < 0.25):
        #             if (momentum_up.iloc[i] and divergence_type == "BULL"):
        #                 axes[3].axvline(x=df["Date"].iloc[i], color='green', linestyle='--', linewidth=1.0)
        #
        #
        #     elif (df["RSI"].iloc[i] <= 30 and df["StochRSI_K"].iloc[i] <= 0.36):
        #             if (momentum_up.iloc[i] and divergence_type == "BULL"):
        #                 axes[3].axvline(x=df["Date"].iloc[i], color='green', linestyle='-', linewidth=1.0)
        #
        #     elif (df["RSI"].iloc[i] <= 30 and df["StochRSI_K"].iloc[i] <= 0.36):
        #             if (momentum_up.iloc[i] and divergence_type == "H-BULL"):
        #                 axes[3].axvline(x=df["Date"].iloc[i], color='green', linestyle='--', linewidth=1.0)

        # Adjust spacing between subplots
        fig.tight_layout()

        # Show the plot for the current coin
        plt.show()





def make_coins():
    symbols = []
    names = []
    symbol_upper = []
    symbol_lower = []
    name_upper = []
    name_lower = []


    num_entries = input("Enter the number of entries: ")
    try:
        num_entries = int(num_entries)
    except ValueError:
        print('--------------------------------------------------------------------')
        print("Invalid input. Please enter a valid number.")
        print('--------------------------------------------------------------------')
        return make_coins()

    for i in range(num_entries):
        print('-------------------------------------------')
        symbol = input("Enter the SYMBOL for entry {}: ".format(i + 1))
        print('-------------------------------------------')
        name = input("Enter the NAME for entry {}: ".format(i + 1))
        print('-------------------------------------------')

        if symbol == "polygon":
            symbol = "matic"
        if name == "matic":
            name = "matic-network"
        else:
            symbol = symbol
            name = name

        print("Symbol:", symbol)
        print("Name:", name)

        symbols.append(symbol)
        names.append(name)
        symbol_upper.append(symbol.upper())
        symbol_lower.append(symbol.lower())
        name_upper.append(name.upper())
        name_lower.append(name.lower())

    data = {
        'symbol': symbols,
        'symbol_upper': symbol_upper,
        'symbol_lower': symbol_lower,
        'name': names,
        'name_upper': name_upper,
        'name_lower': name_lower
    }

    df = pd.DataFrame(data)

    print(df)
    df.to_csv("proto_data.csv")

    valid_names = []
    for name in names:
        url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids={name}&sparkline=false"
        response = requests.get(url)
        if response.status_code != 200 or not response.json():
            print('--------------------------------------------------------------------')
            print("No data was found on coin:", name)
            print('--------------------------------------------------------------------')
        else:
            print("--------------------------------", name, "-- IS VALID!")
            valid_names.append(name)

    df = df[df['name'].isin(valid_names)]  # Remove entries with no data
    print(df)
    df.to_csv("Genesis_data.csv")

    crypt_symbols = df['name_lower'].tolist()
    print(crypt_symbols)
    menu()



    global crypto_symbols
    crypto_symbols = df['name_lower'].tolist()
    print(crypto_symbols)
    menu()


def menu():
    print('--------------------------------------------------------------------')
    print("make graphics - 1")
    print("start over - 2")
    print("get ticker - 3")

    entryz = input("insert option: ")
    if entryz == "1":
        get_charts()
    elif entryz == "2":
        make_coins()
    elif entryz == "3":
        bot_signal()
    else:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("invalid input, try again")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        menu()
menu()




