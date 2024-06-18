from PIL import Image
import glob
#resize delle immagini per il vit
baseheight = 224
width = 224
img_path = 'D:\\ORS\\Data\\Images\\*'
newimg_path = 'D:\\ORS\\Data\\ResizedImages\\'
images = glob.glob(img_path)

for path in images:
    img = Image.open(path)
    img = img.resize((width, baseheight))
    img.save(newimg_path + path[19:])