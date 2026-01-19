import pulp
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def default_constraints():
    return {
        "NB_DAYS": 28,
        "NB_NURSES": 15,
        "MIN_COVERAGE": {
            "Day": 5,
            "Evening": 4,
            "Night": 3
        },
        "MAX_SHIFTS": 20,
        "MAX_NIGHT_SHIFTS": 6,
        "REST_AFTER_NIGHT": True,
        "VACATIONS": {
            "N1": [7, 8, 9, 10],
            "N2": [20, 21, 22, 23]
        }
    }


def ask_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            console.print("[red]Please enter a valid integer.[/red]")

def ask_yes_no(prompt):
    while True:
        v = input(prompt).lower()
        if v in ["y", "yes"]:
            return True
        if v in ["n", "no"]:
            return False
        console.print("[red]Please answer y or n.[/red]")

def ask_days_list(prompt):
    value = input(prompt).strip()
    if value == "":
        return []
    try:
        return [int(x) for x in value.split(",")]
    except ValueError:
        console.print("[red]Invalid format (example: 3,4,10).[/red]")
        return ask_days_list(prompt)



def validate_constraints(cfg):
    if cfg["NB_DAYS"] <= 0 or cfg["NB_NURSES"] <= 0:
        return False, "Days and nurses must be positive."

    for s, v in cfg["MIN_COVERAGE"].items():
        if v <= 0 or v > cfg["NB_NURSES"]:
            return False, f"Invalid coverage for {s}."

    if cfg["MAX_SHIFTS"] <= 0:
        return False, "Max shifts must be positive."

    if cfg["MAX_NIGHT_SHIFTS"] < 0:
        return False, "Max night shifts must be non-negative."

    return True, ""

def manual_constraints():
    while True:
        console.print("\n[bold yellow]Manual constraint entry[/bold yellow]\n")

        cfg = {
            "NB_DAYS": ask_int("Number of days: "),
            "NB_NURSES": ask_int("Number of nurses: "),
            "MIN_COVERAGE": {
                "Day": ask_int("Min nurses for Day shift: "),
                "Evening": ask_int("Min nurses for Evening shift: "),
                "Night": ask_int("Min nurses for Night shift: ")
            },
            "MAX_SHIFTS": ask_int("Max shifts per nurse (month): "),
            "MAX_NIGHT_SHIFTS": ask_int("Max night shifts per nurse: "),
            "REST_AFTER_NIGHT": ask_yes_no("Rest after night shift? (y/n): "),
            "VACATIONS": {}
        }

        console.print("\nEnter vacations (empty nurse name to stop)")
        while True:
            n = input("Nurse name (N1, N2, ...): ").strip()
            if n == "":
                break
            cfg["VACATIONS"][n] = ask_days_list("Days off (e.g. 5,6,7): ")

        valid, msg = validate_constraints(cfg)
        if valid:
            return cfg

        console.print(f"[red]Invalid constraints: {msg}[/red]\n")

def choose_constraints():
    console.print("\n[bold cyan]Nurse Rostering Configuration[/bold cyan]")
    console.print("1 - Use default constraints")
    console.print("2 - Enter constraints manually\n")

    choice = input("Your choice (1 or 2): ").strip()
    return manual_constraints() if choice == "2" else default_constraints()



def solve_model(cfg):
    DAYS = list(range(cfg["NB_DAYS"]))
    NURSES = [f"N{i}" for i in range(1, cfg["NB_NURSES"] + 1)]
    SHIFTS = ["Day", "Evening", "Night"]

    model = pulp.LpProblem("Nurse_Rostering", pulp.LpMinimize)

    x = pulp.LpVariable.dicts("x", (NURSES, DAYS, SHIFTS), cat="Binary")
    max_load = pulp.LpVariable("max_load", lowBound=0, cat="Integer")

    model += max_load

    for d in DAYS:
        for s in SHIFTS:
            model += pulp.lpSum(x[n][d][s] for n in NURSES) >= cfg["MIN_COVERAGE"][s]

    for n in NURSES:
        for d in DAYS:
            model += pulp.lpSum(x[n][d][s] for s in SHIFTS) <= 1

        total = pulp.lpSum(x[n][d][s] for d in DAYS for s in SHIFTS)
        model += total <= cfg["MAX_SHIFTS"]
        model += total <= max_load
        model += pulp.lpSum(x[n][d]["Night"] for d in DAYS) <= cfg["MAX_NIGHT_SHIFTS"]

    model += max_load <= cfg["MAX_SHIFTS"]

    for n, days_off in cfg["VACATIONS"].items():
        if n in NURSES:
            for d in days_off:
                if d in DAYS:
                    for s in SHIFTS:
                        model += x[n][d][s] == 0

    if cfg["REST_AFTER_NIGHT"]:
        for n in NURSES:
            for d in DAYS[:-1]:
                model += (
                    x[n][d]["Night"]
                    + x[n][d+1]["Day"]
                    + x[n][d+1]["Evening"]
                    <= 1
                )

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    return x, NURSES, DAYS, SHIFTS



def display_schedule(x, nurses, days, shifts):
    table = Table(title="Monthly Nurse Schedule", box=box.SIMPLE_HEAVY, show_lines=True)
    table.add_column("Day", justify="center", style="bold")

    for n in nurses:
        table.add_column(n, justify="center")

    colors = {"Day": "cyan", "Evening": "yellow", "Night": "red", "Off": "grey50"}

    for d in days:
        row = [str(d)]
        for n in nurses:
            shift = "Off"
            for s in shifts:
                if x[n][d][s].value() == 1:
                    shift = s
            row.append(f"[{colors[shift]}]{shift}[/{colors[shift]}]")
        table.add_row(*row)

    console.print(table)

def extract_df(x, nurses, days, shifts):
    data = []
    for n in nurses:
        for d in days:
            for s in shifts:
                if x[n][d][s].value() == 1:
                    data.append({"Nurse": n, "Day": d, "Shift": s})
    return pd.DataFrame(data)



def plot_workload(df):
    df.groupby("Nurse").size().plot(kind="bar", title="Workload per Nurse", figsize=(10,5))
    plt.ylabel("Number of shifts")
    plt.tight_layout()
    plt.show()

def plot_shift_distribution(df):
    df["Shift"].value_counts().plot(kind="pie", autopct="%1.1f%%", title="Shift distribution")
    plt.ylabel("")
    plt.show()

def plot_night_balance(df):
    df[df["Shift"] == "Night"].groupby("Nurse").size().plot(
        kind="bar", color="darkred", title="Night shifts per Nurse", figsize=(10,5)
    )
    plt.ylabel("Night shifts")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    cfg = choose_constraints()
    x, NURSES, DAYS, SHIFTS = solve_model(cfg)

    display_schedule(x, NURSES, DAYS, SHIFTS)

    df = extract_df(x, NURSES, DAYS, SHIFTS)

    print("\nGLOBAL STATISTICS\n")
    print(df.groupby("Shift").size())

    print("\nWORKLOAD PER NURSE\n")
    print(df.groupby("Nurse").size())

    print("\nNIGHT SHIFTS PER NURSE\n")
    print(df[df["Shift"] == "Night"].groupby("Nurse").size())

    plot_workload(df)
    plot_shift_distribution(df)
    plot_night_balance(df)
