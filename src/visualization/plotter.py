import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

class DataPlotter:
    def __init__(self, figsize=(20, 5)):
        self.figsize = figsize
        
    def plot_priority(self, df: pd.DataFrame):
        _, ax = plt.subplots(figsize=self.figsize)
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

    def create_model_comparison_plot(self, model_a_results : Dict, model_b_results : Dict, model_names=["Model A", "Model B"]):
        
        class_types = list(model_a_results.keys())
        metrics = ['f1_score', 'accuracy', 'recall']
        
        data = []
        for class_type in class_types:
            for model_name, results in [(model_names[0], model_a_results), (model_names[1], model_b_results)]:
                data.append({
                    'class_type': class_type,
                    'model': model_name,
                    'f1_score': results[class_type]['f1'],
                    'accuracy': results[class_type]['accuracy'],
                    'recall': results[class_type]['recall']
                })
        
        df_comparison = pd.DataFrame(data)
        
        _, axes = plt.subplots(1, len(class_types), figsize=(12, 6))
        if len(class_types) == 1:
            axes = [axes]
        
        colors = ['#3498db', '#e74c3c']
        width = 0.25
        
        for i, class_type in enumerate(class_types):
            ax = axes[i]
            class_data = df_comparison[df_comparison['class_type'] == class_type]
            
            x = np.arange(len(metrics))
            
            for j, model in enumerate([model_names[0], model_names[1]]):
                model_data = class_data[class_data['model'] == model]
                values = [model_data[metric].iloc[0] for metric in metrics]
                
                ax.bar(x + j * width, values, width, label=model, color=colors[j])
            
            ax.set_title(class_type, fontsize=12, fontweight='bold')
            ax.set_xlabel('Classification Metrics')
            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(['F1 Score', 'Accuracy', 'Recall'])
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        return df_comparison
    
    def plot_llm_judge_score(self, ctx : List[int], ans : List[int], grd : List[int], lc : List[bool], title : str):

        fig, ax = plt.subplots(figsize=self.figsize)
        counts = {
            "Context Relevance": [ctx.count('3'), ctx.count('2'), ctx.count('1')],
            "Answer Relevance":  [ans.count('3'), ans.count('2'), ans.count('1')],
            "Groundedness":      [grd.count('3'), grd.count('2'), grd.count('1')],
            "Language Consistency": [lc.count(True), 0, lc.count(False)]
        }
        labels = list(counts.keys())
        score3, score2, score1 = zip(*counts.values())
        x = range(len(labels))
        w = 0.25

        ax.bar(x, score1, w, label="low_relevance / False", color="#F4A261")
        ax.bar(x, score2, w, bottom=score1, label="medium_relevance", color="#A8DADC")
        bottom2 = [i + j for i, j in zip(score1, score2)]
        ax.bar(x, score3, w, bottom=bottom2, label="high_relevance / True", color="#A8E6CF")
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Number of tickets")
        ax.set_title(title)
        ax.legend(title="Score / Is Language Consistent")
        plt.tight_layout()
        plt.show()