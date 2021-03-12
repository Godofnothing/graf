mkdir -p dogs
cd dogs
wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
tar -xvf images.tar
rm images.tar
cd ..