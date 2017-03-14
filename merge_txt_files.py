import glob

def merge_txt_fils(directory, output = 'result.txt'):

  read_files = glob.glob(str(directory) + "/*.txt")

  with open(output, "wb") as outfile:
      for f in read_files:
          with open(f, "rb") as infile:
              outfile.write(infile.read())

if __name__ == '__main__':
  directories = ['C:/Users/Wojtek/OneDrive/ubuntu_shared/sony/bioNLP/data/pc13/',
                 'C:/Users/Wojtek/OneDrive/ubuntu_shared/sony/bioNLP/data/ge11/',
                 'C:/Users/Wojtek/OneDrive/ubuntu_shared/sony/bioNLP/data/all/']

  for directory in directories:
    merge_txt_fils(directory, directory + '/result.txt')