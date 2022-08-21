select
	Name
from STUDENTS
where Marks > 75
order by lower(RIGHT(Name,3)) asc, ID asc;