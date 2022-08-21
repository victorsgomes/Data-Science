select 
	concat(Name,'(',LEFT(Occupation,1),')') as output
from OCCUPATIONS
order by Name

select
	concat('There are a total of ',count(Occupation),' ',lower(Occupation),'s.')
from OCCUPATIONS
group by Occupation
order by count(Occupation), Occupation;	