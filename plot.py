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

    idx = 0

    for name, records in kwargs.items():

        y = records[1]
        x = [records[0] * i for i in range(len(y))]
        data_count = len(y)

        # z1 = np.polyfit(x, y, 10) # 用3次多项式拟合
        # p1 = np.poly1d(z1)

        # yvals=p1(x)
        # 也可以使用yvals=np.polyval(z1,x)
        x_smooth = np.linspace(min(x), max(x), data_count * smooth_ration)
        y_smooth = interpolate.spline(x, y, x_smooth)

        # tck = interpolate.spline(x, y)
        plt.plot(x, y, "-", label=name, linewidth=2.5)
        plt.minorticks_on()
        plt.grid(which="major", color="gray", linestyle="-", linewidth=1)
        plt.grid(which="minor", color="gray", linestyle=":", linewidth=0.5)


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

file_name1 = r"C:\Users\Gloria\Desktop\实验结果对比\BY_0423_1115.BbB44.json"
file_name2 = r"C:\Users\Gloria\Desktop\实验结果对比\BY_0420_1128.BbBgaussian7.json"
file_name3 = r"C:\Users\Gloria\Desktop\实验结果对比\DROP_0419_2033.SGD.json"
file_name4 = r"C:\Users\Gloria\Desktop\实验结果对比\DROP_0419_1529.Dropout.json"
file_name5 = r"C:\Users\Gloria\Desktop\实验结果对比\DROP_0419_1645.Dropconnect1.json"
file2_name = r"keeps\sigmoid_changing\fixed_back_coffe\alpha20_center015_upper006_coeff_{}.json"

# for_(key, file_name)


accu = {

    "BbB": for_("valid_accu", file_name1),
    "BbB_gaussian": for_("valid_accu", file_name2),
    "SGD": for_("valid_accu", file_name3),
    "Dropout": for_("valid_accu", file_name4),
    "Dropconnect": for_("valid_accu", file_name5),

}


curve_graph(**accu)

