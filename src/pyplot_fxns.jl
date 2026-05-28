module pyplot_fxns
using PyCall, PyPlot
mc = PyPlot.matplotlib.colors
@pyimport colorsys as cs


function remove_axis_box(ax; s=["top", "right"])
    """ remove subplot axis box sides
    """
    for i in s        
        ax.spines[i].set_visible(false)
        if i == "bottom"
            ax.set_xticks([])
        elseif i == "left"
            ax.set_yticks([])
        end
    end
return ax
end


function remove_axis_box_t(ax; s=["top", "right"], bottom_ticklabels=false, left_ticklabels=false)
    """ remove subplot axis box sides and ticklabels
    """
    for i in s        
        ax.spines[i].set_visible(false)
        if i == "bottom"
            if bottom_ticklabels == false
                ax.set_xticklabels([])
            end
        elseif i == "left"
            if left_ticklabels ==false
                ax.set_yticklabels([])
            end
        end
    end
return ax
end


function lighten_color(color, amount::Real=0.5)
    """
    lighten_color(color, amount=0.5)

    Lightens the given color by multiplying (1 − luminosity) by `amount`.

    `color` can be:
    - a matplotlib color string (e.g. `"g"`)
    - a hex string (e.g. `"#F034A3"`)
    - an RGB tuple or vector (e.g. `(0.3, 0.55, 0.1)`)

    Returns an RGB tuple.
    """
    # Resolve named colors if possible
    c = try
        mc.cnames[color]
    catch
        color
    end

    # Convert to RGB via matplotlib
    rgb = mc.to_rgb(c)

    # Convert RGB → HLS
    h, l, s = cs.rgb_to_hls(rgb...)

    # Lighten
    new_l = 1 - amount * (1 - l)

    # Convert back to RGB
    return Tuple(cs.hls_to_rgb(h, new_l, s))
end

end