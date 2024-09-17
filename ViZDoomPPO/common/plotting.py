import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_activations(hooks):
    f = plt.figure(constrained_layout=False, figsize=(12, 8))
    gs = f.add_gridspec(3, 3)

    ax = [f.add_subplot(gs[0, :2]), f.add_subplot(gs[1, :2]), f.add_subplot(gs[2, :2])]
    ax_hists = [f.add_subplot(gs[0, 2]), f.add_subplot(gs[1, 2]), f.add_subplot(gs[2, 2])]

    ax[0].set_title('Layer activation mean')
    ax[1].set_title('Layer activation standard deviation')
    ax[2].set_title('Low activation proportion')

    for i, h in enumerate(hooks):
        stacked_data = np.stack(h.activation_data)

        # Compute statistics
        means = np.mean(stacked_data, axis=(1, 2, 3, 4))
        stds = np.std(stacked_data, axis=(1, 2, 3, 4))
        low_act = ((-0.2 <= stacked_data) & (stacked_data <= 0.2))
        low_act = np.count_nonzero(low_act, axis=(1, 2, 3, 4)) / np.prod(low_act.shape[1:])

        # Histograms
        bins = np.linspace(-7, 7, 40)
        melted_data = stacked_data.reshape(stacked_data.shape[0], -1)
        hist_img = np.apply_along_axis(
            lambda a: np.log1p(np.histogram(a, bins=bins)[0][::-1]), 1, melted_data)

        # Plot
        ax[0].plot(means, label=f'Mean layer {i}')
        ax[1].plot(stds, label=f'Std layer {i}')
        ax[2].plot(low_act, label=f'Low activation layer {i}')
        ax_hists[i].imshow(hist_img.T, aspect='auto')
        ax_hists[i].set_title(f'Activation histogram layer {i}')

    ax[0].set_ylim((-0.5, 0.5))
    ax[1].set_ylim((0, 1))
    ax[2].set_ylim((0, 1))

    for a in ax:
        a.grid(True)
        a.legend()

    plt.tight_layout()


def plot_evaluation_results(path: str):
    evaluations = np.load(path)
    ts = evaluations['timesteps']
    results = evaluations['results'].mean(axis=1).squeeze()

    f, ax = plt.subplots(1, 1, figsize=(8, 5))
    ticker = matplotlib.ticker.EngFormatter()
    ax.xaxis.set_major_formatter(ticker)
    ax.xaxis.set_minor_formatter(ticker)

    ax.plot(ts, results)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Mean reward')

    ax.grid(True)