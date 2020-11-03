cd ~/
perl -v
pip install gdown
gdown https://drive.google.com/uc?id=1i-K3TddTOffBC85gWh0sXxUad_cGXl1Z
unzip rouge.zip

cd rouge
tar -zxvf XML-Parser-2.44.tar.gz
cd XML-Parser-2.44
perl Makefile.PL 
make 
make test
sudo make install  

cd ..
tar -zxvf XML-RegExp-0.04.tar.gz
cd XML-RegExp-0.04
perl Makefile.PL 
make 
make test 
sudo make install

sudo apt-get update
sudo apt-get install libwww-perl

sudo apt-get install libxml-perl

cd ..
tar -zxvf XML-DOM-1.46.tar.gz
cd XML-DOM-1.46
perl Makefile.PL 
make 
make test 
sudo make install 

sudo apt-get install libdb-dev
cd ..
tar -zxvf DB_File-1.835.tar.gz
cd DB_File-1.835
perl Makefile.PL 
make 
make test 
sudo make install

cd ..
tar -zxvf ROUGE-1.5.5.tgz

echo 'export ROUGE_EVAL_HOME="$ROUGE_EVAL_HOME:/home/ubuntu/rouge/RELEASE-1.5.5/data"' >> ~/.profile
source ~/.profile
