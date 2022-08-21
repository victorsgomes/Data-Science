select distinct
	CITY
from STATION
where lower(substring(CITY,1,1)) in ('a','e','i','o','u')