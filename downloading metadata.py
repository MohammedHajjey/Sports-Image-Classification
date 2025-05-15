import kagglehub

# Download latest version
path = kagglehub.dataset_download("gpiosenka/sports-classification")

print("Path to dataset files:", path)


import shutil
destenation_path = 'C:/Users/User/Desktop/Universty Document/term8/Bulut-bilişim-Yapayzeka/Sports-Image-Classification/'
shutil.move(path, destenation_path)