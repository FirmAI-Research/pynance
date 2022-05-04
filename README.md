http://34.207.78.246/treasury_rates/
http://ec2-34-207-78-246.compute-1.amazonaws.com:8000/treasury_rates/


***pystan & fbprophet ***
windows (run conda prompt as admim)
conda install numpy cython -c conda-forge
conda install pystan -c conda-forge
conda install fbprophet
...
https://stackoverflow.com/questions/53178281/installing-fbprophet-python-on-windows-10



--
pip install pystan~=2.14

linux
--conda
cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
bash Anaconda3-2021.05-Linux-x86_64.sh
... use -u flag to update existing install

source ~/.bashrc
conda list
conda create --name my_env python=3

conda install -c conda-forge pystan
conda install -c conda-forge fbprophet

--ec2: OSError: [Errno 28] No space left on device
df -h
lsblk
df -hT   ---> /dev/root
sudo resize2fs /dev/root
<Reboot ec2 instance after adding storage to volume>

***wip***
django hosted with apache on ubuntu ec2<br>
cloudtables via https://datatables.net/ (datables)<br>
highcharts.js<br>

***todo***
python logging<br>
sqlite in dev --> aws rds in prod<br>
react on the frontend<br>
add css for custom stylesheets

***useful***

history | grep ...[recent cmd name]...
vim --> esc :%d; i :w  :x  :q!


***Apache***

sudo apt install letsencrypt
sudo apt install certbot python3-certbot-apache
sudo a2enmod proxy
sudo a2enmod proxy_http
sudo a2enmod proxy_balancer
sudo a2enmod lbmethod_byrequests 
sudo a2enmod ssl

[/etc/apache2/sites-available/000-default.conf]
<VirtualHost *:80>
        <!-- ServerName example.com
        ServerAlias www.example.com
        ServerAdmin webmaster@example.com
        ErrorLog ${APACHE_LOG_DIR}/error.log
        CustomLog ${APACHE_LOG_DIR}/access.log combined -->

        ProxyRequests Off
        <Proxy *>
          Order deny,allow
          Allow from all
        </Proxy>
        
        ProxyPass / http://0.0.0.0:8000/
        ProxyPassReverse / http://0.0.0.0:8000/

        <Location />
          Order allow,deny
          Allow from all
        </Location>

</VirtualHost>



apachectl configtest

---
nohup python manage.py runserver 0.0.0.0:8000 &
top 
kill pid --##--
re run nohup

sudo apt install tree