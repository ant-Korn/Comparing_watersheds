import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from matplotlib import rc
from collections import OrderedDict
import argparse

plt.rcParams['font.size'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
rc('text', usetex=True)
rc('text.latex',unicode=True)
rc('text.latex',preamble='\\usepackage[utf8]{inputenc}')
rc('text.latex',preamble='\\usepackage[english]{babel}')

titles = {'opencv': 'OpenCV', 'mahotas': 'Mahotas', 'mamba': 'Mamba', 'smil': 'SMIL', 'skimage': 'Scikit-image', 'itk': 'ITK'}

parser = argparse.ArgumentParser(description='Make memory consumption plots.')
parser.add_argument("-lg", "--logarifmic", help="Logarifmic scale.", action='store_true')
parser.add_argument("-wl", "--watershed_lines", help="With watershed lines.", action='store_true')
parser.add_argument("-d", "--dimansions", help="Number of image dimansions.", type=int)
args = parser.parse_args()

LOGARIFMIC = args.logarifmic
WITH_WL = args.watershed_lines
D = args.dimansions

if WITH_WL:
    str_wl = ' (with {WL})'
else: 
    str_wl = ' (without {WL})'

if D == 2:
    needen_sizes = [512, 1861, 2581, 3140, 3614, 4033 ]
elif D == 3:
    needen_sizes = [64, 151, 188, 214, 236, 253]

if __name__ == "__main__":
    if D == 2:
        if WITH_WL:
            df = pd.read_csv("logs_watershed/csv/proc_mem_WL.csv", header=[0,1], index_col=0)
        else:
            df = pd.read_csv("logs_watershed/csv/proc_mem.csv", header=[0,1], index_col=0)
    elif D == 3:
        if WITH_WL:
            df = pd.read_csv("logs_watershed/3D/csv/3D_proc_mem_WL.csv", header=[0,1], index_col=0)
        else:
            df = pd.read_csv("logs_watershed/3D/csv/3D_proc_mem.csv", header=[0,1], index_col=0)
    if WITH_WL:
        level0 = [item[0] for item in df.columns.get_values()]
    else:
        level0 = [item[0] for item in df.columns.get_values() if item[0] != 'opencv']
    level0 = list(OrderedDict.fromkeys(level0))
    level1 = list({item[1] for item in df.columns.get_values()})
    if WITH_WL and D == 2:
        level0.insert(len(level0), level0.pop(3))
    titles_l0 = [titles[level] for level in level0]
    clusters = list(OrderedDict.fromkeys(df[level0[0],'img_name']))
    print(clusters)
    print(level0, level1)

    for cluster in clusters:
        plt.figure()
        ax = plt.gca()
        for lib in level0:
            table = df[lib].loc[df[lib]['img_name'] == cluster]
            if D == 2:
                ticks = tuple(zip(*[(sz*sz, str(sz)+'x'+str(sz)) for sz in table['size']
                                                        if sz in needen_sizes]))
                table['size'] = table['size'] ** 2
            elif D == 3:
                ticks = tuple(zip(*[(sz*sz*sz, str(sz)+'x'+str(sz)+'x'+str(sz)) for sz in table['size']
                                                        if sz in needen_sizes]))
                table['size'] = table['size'] ** 3
            table.plot(x='size', y='memory', style='--', marker='.', grid=True, ax=ax, logy=LOGARIFMIC)
        ax.legend(titles_l0)
        if D == 2:
            ax.set_xlabel('Image size, pix')
        elif D == 3:
            ax.set_xlabel('Image size, vox')
        plt.xticks(ticks[0], ticks[1])
        if LOGARIFMIC:
            time_unit = 'MiB (logarithmic scale)'
        else:
            time_unit = 'MiB'
        ax.set_ylabel('Peak memory size, '+time_unit)
        if cluster == 'circles':
            cluster = 'cells'
        if LOGARIFMIC:
            title_plot = cluster
        else:
            title_plot = cluster
        title_plot = title_plot.capitalize()
        title_plot += str_wl
        ax.set_title(title_plot)
        if WITH_WL:
            plt.savefig('plots/'+cluster+'memory_WL.pdf',bbox_inches='tight')
        else:
            plt.savefig('plots/'+cluster+'memory.pdf',bbox_inches='tight')
