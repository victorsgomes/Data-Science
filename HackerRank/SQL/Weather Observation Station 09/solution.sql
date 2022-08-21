select distinct
	CITY
from STATION
where lower(LEFT(CITY,1)) not in ('a','e','i','o','u');	