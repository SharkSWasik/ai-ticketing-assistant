import pandas as pd
import matplotlib.pyplot as plt

class DataPlotter:
    def __init__(self, figsize=(20, 5)):
        self.figsize = figsize
        
    def plot_priority(self, ax: plt.Axes, df: pd.DataFrame):
        counts = df['priority'].value_counts().reindex(['high','medium','low'])
        counts.plot(
            kind="bar",
            color=['#ff9999', '#ffff99', '#ffcc99'],
            ax=ax
        )
        ax.set_title("Number of tickets per priority")
        ax.set_ylabel("Number of tickets")
    
    def plot_language(self, ax: plt.Axes, df: pd.DataFrame):
        counts = df.groupby(['language','priority']).size().unstack(fill_value=0)
        counts[['high','medium','low']].plot(
            kind="bar",
            stacked=True,
            color=['#ff9999', '#ffff99', '#ffcc99'],
            ax=ax
        )
        ax.set_title("Number of tickets per language")
        ax.set_ylabel("Number of tickets")

    def plot_support_team(self, ax: plt.Axes, df: pd.DataFrame):
        queues_order = [
            'Billing and Payments', 'Customer Service', 'General Inquiry',
            'Human Resources', 'IT Support', 'Product Support',
            'Returns and Exchanges', 'Sales and Pre-Sales',
            'Service Outages and Maintenance', 'Technical Support'
        ]
        counts = df['queue'].value_counts().reindex(queues_order).sort_values()
        counts.plot(kind="barh", colormap="Set2", ax=ax)
        ax.set_title("Number of tickets per queue")
        ax.set_xlabel("Number of tickets")

    def plot_ticket_overview(self, df: pd.DataFrame):
        """
        Creates a 1×3 grid of bar charts by delegating to helper methods.
        """
        _, axs = plt.subplots(1, 3, figsize=self.figsize)

        self.plot_priority(axs[0], df)
        self.plot_language(axs[1], df)
        self.plot_support_team(axs[2], df)

        plt.tight_layout()
        plt.show()

    def plot_metric_scores(self, df_score: pd.DataFrame):
        _, ax = plt.subplots(figsize=(7, 4))
        df_score.plot(x="class_type", y=["f1 score", "accuracy", "recall"], kind="bar", ax=ax, rot=0, ylim=(0, 1))
        ax.set_title("Classification Performance Metrics")
        plt.tight_layout()
        plt.show()