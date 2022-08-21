select 
	round(LAT_N,4)
from STATION
where LAT_N > 38.7780
order by LAT_N asc
limit 1;