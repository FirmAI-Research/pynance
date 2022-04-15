http://34.207.78.246/treasury_rates/
http://ec2-34-207-78-246.compute-1.amazonaws.com:8000/treasury_rates/


***wip***
currently django full stack hosted with apache on ubuntu ec2<br>
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