import matplotlib.pyplot as plt
import numpy as np

from scipy import interpolate

from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)


def fit_example():
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(1, 17, 1)
    y = np.array(
        [
            4.00,
            6.40,
            8.00,
            8.80,
            9.22,
            9.50,
            9.70,
            9.86,
            10.00,
            10.20,
            10.32,
            10.42,
            10.50,
            10.55,
            10.58,
            10.60,
        ]
    )
    z1 = np.polyfit(x, y, 3)  # 用3次多项式拟合
    p1 = np.poly1d(z1)

    yvals = p1(x)  # 也可以使用yvals=np.polyval(z1,x)
    plot1 = plt.plot(x, y, "*", label="original values")
    plot2 = plt.plot(x, yvals, "r", label="polyfit values")
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.legend(loc=4)  # 指定legend的位置,读者可以自己help它的用法
    plt.title("polyfitting")
    plt.show()
    plt.savefig("p1.png")


from mtrain.watcher import parse_losses_record, parse_watcher_dict


# ================    ===============================
# character           description
# ================    ===============================
#    -                solid line style
#    --               dashed line style
#    -.               dash-dot line style
#    :                dotted line style
#    .                point marker
#    ,                pixel marker
#    o                circle marker
#    v                triangle_down marker
#    ^                triangle_up marker
#    <                triangle_left marker
#    >                triangle_right marker
#    1                tri_down marker
#    2                tri_up marker
#    3                tri_left marker
#    4                tri_right marker
#    s                square marker
#    p                pentagon marker
#    *                star marker
#    h                hexagon1 marker
#    H                hexagon2 marker
#    +                plus marker
#    x                x marker
#    D                diamond marker
#    d                thin_diamond marker
#    |                vline marker
#    _                hline marker
# ================    ===============================


def curve_graph(smooth_ration=10, **kwargs):

    color = ["red", "green", "red"]
    idx = 0

    for name, records in kwargs.items():

        y = [0,] + records[1]
        x = [records[0] * i for i in range(len(y))]
        data_count = len(y)

        # z1 = np.polyfit(x, y, 10) # 用3次多项式拟合
        # p1 = np.poly1d(z1)

        # yvals=p1(x)
        # 也可以使用yvals=np.polyval(z1,x)
        x_smooth = np.linspace(min(x), max(x), data_count * smooth_ration)
        y_smooth = interpolate.spline(x, y, x_smooth)

        # tck = interpolate.spline(x, y)
        plt.plot(x, y, "-", label=name, linewidth=2.5, color=color[idx])
        plt.minorticks_on()
        plt.grid(which="major", color="gray", linestyle="-", linewidth=1)
        plt.grid(which="minor", color="gray", linestyle=":", linewidth=0.5)
        plt.axvline(x=300, ymin=0, ymax=1, linestyle="--", linewidth=2)
        plt.axvline(x=500, ymin=0, ymax=1, linestyle="--", linewidth=2)
        plt.axvline(
            x=2000,
            ymin=0,
            ymax=1,
            linestyle="--",
            linewidth=2,
            color="gray",
        )
        # plt.axvline(x=2.20589566)
        idx += 1

    plt.legend(loc="best")
    plt.title("A10 to W10+10")
    plt.show()


def for_(name, file):
    record_dic = parse_watcher_dict(file)
    losses = parse_losses_record(record_dic)
    return losses[name]


def for_accu(file):
    record_dic = parse_watcher_dict(file)
    losses = parse_losses_record(record_dic)
    return losses["valid_accu"]


def for_bias(file):
    record_dic = parse_watcher_dict(file)
    losses = parse_losses_record(record_dic)
    return losses["bias"]


# def bias(p, alpha=20, center=0.2, high=0.06):

#     z = (
#         (
#             1 / (1 + np.exp(-alpha * (p - center)))
#             - 1 / (1 + np.exp(-alpha * (-center)))
#         )
#         * ((1 + np.exp(alpha * center)) / np.exp(alpha * center))
#         * high
#     )

#     return z


# x = [i / 10000 for i in range(10000)]
# y = [bias(xi) for xi in x]


# plt.plot(x, y, "-", linewidth=2.5)
# plt.show()

# assert False

file_name = r"RECORDS\Pr2Cl.json"

file2_name = r"keeps\sigmoid_changing\fixed_back_coffe\alpha20_center015_upper006_coeff_{}.json"

accu = {
    # "1": for_('tolorate', file_name.format(1)),
    "outlier ": for_('outlier_data', file_name.format(3)),
    "valid ": for_("valid_data", file_name),
    # "outlier": for_('outlier_data', file_name.format(3)),
    # "3": for_('valid_accu', file_name.format(3)),
    # "4": for_('valid_accu', file_name.format(4)),
    # "11": for_("outlier_data", file_name),
    # "12": for_("valid_accu", file_name.format(2)),
}


curve_graph(**accu)

