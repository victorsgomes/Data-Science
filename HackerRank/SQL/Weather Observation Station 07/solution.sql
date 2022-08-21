select distinct 
	CITY
from STATION
where lower(RIGTH(CITY,1)) in ('a','e','i','o','u');