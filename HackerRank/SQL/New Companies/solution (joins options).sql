select
	concat(LEFT(company_code,1),LPAD(SUBSTRING(company_code,2,99),(select max(char_length(company_code)) from Company) - 1,'0')) as ordenate,
	company_code,
	founder
from Company
order by ordenate;

/*CONTAGEM DAS TABELAS*/

select
	count(lUni.lead_manager_code) as count_lead_manager_code,
	lUni.company_code as lead_company_code
from (select distinct lead_manager_code, company_code from Lead_Manager) as lUni
group by lead_company_code;

select 
	count(sUni.senior_manager_code) as count_senior_manager_code,
	sUni.company_code as senior_company_code
from (select distinct senior_manager_code, company_code from Senior_Manager) as sUni 
group by senior_company_code;	

select 
	count(mUni.manager_code) as count_manager_code,
	mUni.company_code as manager_company_code
from (select distinct manager_code, company_code from Manager) as mUni
group by manager_company_code;

select 
	count(eUni.employee_code) as count_employee_code,
	eUni.company_code as employee_company_code
from (select distinct employee_code, company_code from Employee)  as eUni
group by employee_company_code;	


/*JOINS*/
/*1*/
select
	c.company_code,
	c.founder,
	l.lead_company_code,
	l.count_lead_manager_code
from Company as c
left join (select count(lUni.lead_manager_code) as count_lead_manager_code, lUni.company_code as lead_company_code
		from (select distinct lead_manager_code, company_code from Lead_Manager) as lUni
		group by lead_company_code) as l on c.company_code = l.lead_company_code;

/*2*/

select 
	tcOne.company_code,
	tcOne.founder,
	tcOne.count_lead_manager_code,
	s.count_senior_manager_code,
	s.senior_company_code
from (select
			c.company_code,
			c.founder,
			l.lead_company_code,
			l.count_lead_manager_code
		from Company as c
		left join (select count(lUni.lead_manager_code) as count_lead_manager_code, lUni.company_code as lead_company_code
				from (select distinct lead_manager_code, company_code from Lead_Manager) as lUni
				group by lead_company_code) as l on c.company_code = l.lead_company_code) as tcOne
left join (select 
				count(sUni.senior_manager_code) as count_senior_manager_code,
				sUni.company_code as senior_company_code
			from (select distinct senior_manager_code, company_code from Senior_Manager) as sUni 
			group by senior_company_code) as s on tcOne.company_code = s.senior_company_code;					

/*3*/

select
	tcTwo.company_code,
	tcTwo.founder,
	tcTwo.count_lead_manager_code,
	tcTwo.count_senior_manager_code,
	m.count_manager_code,
	m.manager_company_code
from (select 
			tcOne.company_code,
			tcOne.founder,
			tcOne.count_lead_manager_code,
			s.count_senior_manager_code,
			s.senior_company_code
		from (select
					c.company_code,
					c.founder,
					l.lead_company_code,
					l.count_lead_manager_code
				from Company as c
				left join (select count(lUni.lead_manager_code) as count_lead_manager_code, lUni.company_code as lead_company_code
						from (select distinct lead_manager_code, company_code from Lead_Manager) as lUni
						group by lead_company_code) as l on c.company_code = l.lead_company_code) as tcOne
		left join (select 
						count(sUni.senior_manager_code) as count_senior_manager_code,
						sUni.company_code as senior_company_code
					from (select distinct senior_manager_code, company_code from Senior_Manager) as sUni 
					group by senior_company_code) as s on tcOne.company_code = s.senior_company_code) as tcTwo
left join (select 
				count(mUni.manager_code) as count_manager_code,
				mUni.company_code as manager_company_code
			from (select distinct manager_code, company_code from Manager) as mUni
			group by manager_company_code) as m on tcTwo.company_code = m.manager_company_code;					

/*4*/

select
	tcThree.company_code,
	tcThree.founder,
	tcThree.count_lead_manager_code,
	tcThree.count_senior_manager_code,
	tcThree.count_manager_code,
	e.count_employee_code,
	e.employee_company_code
from (select
			tcTwo.company_code,
			tcTwo.founder,
			tcTwo.count_lead_manager_code,
			tcTwo.count_senior_manager_code,
			m.count_manager_code,
			m.manager_company_code
		from (select 
					tcOne.company_code,
					tcOne.founder,
					tcOne.count_lead_manager_code,
					s.count_senior_manager_code,
					s.senior_company_code
				from (select
							c.company_code,
							c.founder,
							l.lead_company_code,
							l.count_lead_manager_code
						from Company as c
						left join (select count(lUni.lead_manager_code) as count_lead_manager_code, lUni.company_code as lead_company_code
								from (select distinct lead_manager_code, company_code from Lead_Manager) as lUni
								group by lead_company_code) as l on c.company_code = l.lead_company_code) as tcOne
				left join (select 
								count(sUni.senior_manager_code) as count_senior_manager_code,
								sUni.company_code as senior_company_code
							from (select distinct senior_manager_code, company_code from Senior_Manager) as sUni 
							group by senior_company_code) as s on tcOne.company_code = s.senior_company_code) as tcTwo
		left join (select 
						count(mUni.manager_code) as count_manager_code,
						mUni.company_code as manager_company_code
					from (select distinct manager_code, company_code from Manager) as mUni
					group by manager_company_code) as m on tcTwo.company_code = m.manager_company_code) as tcThree
left join (select 
				count(eUni.employee_code) as count_employee_code,
				eUni.company_code as employee_company_code
			from (select distinct employee_code, company_code from Employee) as eUni
			group by employee_company_code) as e on tcThree.company_code = e.employee_company_code;

/*CRIANDO ORDENADOR*/

select
	tcOrder.company_code,
	tcOrder.founder,
	tcOrder.count_lead_manager_code,
	tcOrder.count_senior_manager_code,
	tcOrder.count_manager_code,
	tcOrder.count_employee_code
from (select
			tcThree.company_code,
			tcThree.founder,
			tcThree.count_lead_manager_code,
			tcThree.count_senior_manager_code,
			tcThree.count_manager_code,
			e.count_employee_code,
			e.employee_company_code
		from (select
					tcTwo.company_code,
					tcTwo.founder,
					tcTwo.count_lead_manager_code,
					tcTwo.count_senior_manager_code,
					m.count_manager_code,
					m.manager_company_code
				from (select 
							tcOne.company_code,
							tcOne.founder,
							tcOne.count_lead_manager_code,
							s.count_senior_manager_code,
							s.senior_company_code
						from (select
									c.company_code,
									c.founder,
									l.lead_company_code,
									l.count_lead_manager_code
								from Company as c
								left join (select count(lUni.lead_manager_code) as count_lead_manager_code, lUni.company_code as lead_company_code
										from (select distinct lead_manager_code, company_code from Lead_Manager) as lUni
										group by lead_company_code) as l on c.company_code = l.lead_company_code) as tcOne
						left join (select 
										count(sUni.senior_manager_code) as count_senior_manager_code,
										sUni.company_code as senior_company_code
									from (select distinct senior_manager_code, company_code from Senior_Manager) as sUni 
									group by senior_company_code) as s on tcOne.company_code = s.senior_company_code) as tcTwo
				left join (select 
								count(mUni.manager_code) as count_manager_code,
								mUni.company_code as manager_company_code
							from (select distinct manager_code, company_code from Manager) as mUni
							group by manager_company_code) as m on tcTwo.company_code = m.manager_company_code) as tcThree
		left join (select 
						count(eUni.employee_code) as count_employee_code,
						eUni.company_code as employee_company_code
					from (select distinct employee_code, company_code from Employee) as eUni
					group by employee_company_code) as e on tcThree.company_code = e.employee_company_code) as tcOrder
order by tcOrder.company_code;								