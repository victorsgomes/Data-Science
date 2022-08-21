select distinct
	min(CITY),
	length(CITY)
from STATION
group by length(CITY)
order by length(CITY);