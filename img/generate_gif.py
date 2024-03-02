from PIL import Image
import glob

gif_path = "./img/"
fig_list = sorted(glob.glob(gif_path+"*.png"),key=lambda name:int(name[len(gif_path):-4]))

imageDim = []

for f in range(len(fig_list)):
    imageDim.append(Image.open(fig_list[f]))
    for i in range(30):
        imageDim.append(Image.open(fig_list[f]))
print(len(imageDim))
# duration = [0.001]*len(fig_list)
# duration[-1] = 1

imageDim[0].save("./Bayesian_Optimization2d.gif",
                 save_all=True,
                 append_images=imageDim,
                 duration=20,
                 loop=False)
