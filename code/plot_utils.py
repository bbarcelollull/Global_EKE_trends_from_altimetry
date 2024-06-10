import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as feature

ccrs_land = feature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='black',
                                        facecolor='navajowhite',
                                        linewidth=0.2)


def map_plot(ds_list,
             vmin=0,
             vmax=100,
             cmap='jet',
             titles='',
             label='',
             coords=[0.1, 359.8, -65, 65]):

    fig = plt.figure(figsize=(10, 5), dpi=200)

    gs = mpl.gridspec.GridSpec(len(ds_list), 1, figure=fig)

    for i in range(len(ds_list)):

        ax = fig.add_subplot(gs[i, 0],
                             projection=ccrs.Robinson(central_longitude=(coords[1]-coords[0])/2))

        ds_list[i].plot(transform=ccrs.PlateCarree(),
                        cbar_kwargs=dict(label=label),
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        ax=ax)
        ax.set_extent([coords[0], coords[1], coords[2],
                       coords[3]], crs=ccrs.PlateCarree())

        if titles == '':
            titles = [titles]*len(ds_list)

        ax.set_title(titles[i], fontsize=7)

        ax.add_feature(ccrs_land)

    return



def plot_bar(x, y, ax, wdth, text_size, fs, text_color='k', nums=None, hatchs = None, kind='all_sat'):#,  nums = None):
        if kind == 'two_sat':
            c = 'cornflowerblue'
            ec = 'royalblue'
            zo = 1
        elif kind == 'all_sat':
            c = 'lightcoral'
            ec = 'indianred'
            zo = 2

        else:
            c = 'mediumseagreen'
            ec = 'seagreen'
            zo = 3
            
        ax.bar(x, height=y, width=wdth, align='center',
               color=c, edgecolor=ec,hatch=hatchs, zorder=zo)

        if nums is not None:
            for i, num in enumerate(nums):
                if y[i]>=0:
                    ax.text(x[i], y[i]+text_size*fs/20, f'{num:.01f}%',
                        fontsize=4*text_size*fs-2, color=text_color,
                        ha='center', va='bottom')
                elif y[i]<0:
                    ax.text(x[i], 0.08, f'{num:.01f}%',
                        fontsize=4*text_size*fs-2, color=text_color,
                        ha='center', va='bottom')
                    


