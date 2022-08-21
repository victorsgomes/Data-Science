select distinct
	CITY
from STATION
where (lower(LEFT(CITY,1)) in ('a','e','i','o','u')) and (lower(RIGHT(CITY,1)) in ('a','e','i','o','u'));