select
	colle.college_id as college_id,
	colle.contest_id as colle_contest_id,
	cont.contest_id as cont_contest_id,
	cont.hacker_id as hacker_id,
	cont.name as name
from Colleges as colle
left join Contests as cont on colle.contest_id = cont.contest_id;

/*CONCATENANDO COM A TABELA DOS CHALLENGES*/

select 
	chal.challenge_id as chal_challenge_id,
	chal.college_id as chal_college_id,
	concat.college_id as concat_college_id,
	concat.colle_contest_id as concat_contest_id,
	concat.hacker_id as hacker_id,
	concat.name as name	
from (select
		colle.college_id as college_id,
		colle.contest_id as colle_contest_id,
		cont.contest_id as cont_contest_id,
		cont.hacker_id as hacker_id,
		cont.name as name
	from Colleges as colle
	left join Contests as cont on colle.contest_id = cont.contest_id) as concat
left join Challenges as chal on concat.college_id = chal.college_id;	

/*AGRUPANDO AS TABELAS DE CONTAGEM*/

select
	challenge_id as view_id,
	sum(total_views),
	sum(total_unique_views)
from View_Stats
group by challenge_id;

select
	challenge_id as sub_id,
	sum(total_submissions),
	sum(total_accepted_submissions)
from Submission_Stats
group by challenge_id;	

/*CONCATENANDO AS TABELAS DE CONTAGEM*/

select
	v.view_id,
	v.sum_total_views,
	v.sum_total_unique_views,
	s.sub_id,
	s.sum_total_submissions,
	s.sum_total_accepted_submissions
from (select
		challenge_id as view_id,
		sum(total_views) as sum_total_views,
		sum(total_unique_views) as sum_total_unique_views
	from View_Stats
	group by challenge_id)	as v
left join (select
			challenge_id as sub_id,
			sum(total_submissions) as sum_total_submissions,
			sum(total_accepted_submissions) as sum_total_accepted_submissions
		from Submission_Stats
		group by challenge_id) as s on s.sub_id = v.view_id
UNION
select
	v.view_id,
	v.sum_total_views,
	v.sum_total_unique_views,
	s.sub_id,
	s.sum_total_submissions,
	s.sum_total_accepted_submissions
from (select
		challenge_id as view_id,
		sum(total_views) as sum_total_views,
		sum(total_unique_views) as sum_total_unique_views
	from View_Stats
	group by challenge_id)	as v
right join (select
			challenge_id as sub_id,
			sum(total_submissions) as sum_total_submissions,
			sum(total_accepted_submissions) as sum_total_accepted_submissions
		from Submission_Stats
		group by challenge_id) as s on s.sub_id = v.view_id;


/*ALTERANDO OS NULL POR 0*/
select 
 	cont.view_id as view_id,
 	coalesce(cont.sum_total_views,0) as sum_total_views,
 	coalesce(cont.sum_total_unique_views,0) as sum_total_unique_views,
 	coalesce(cont.sum_total_submissions,0) as sum_total_submissions,
 	coalesce(cont.sum_total_accepted_submissions,0) as sum_total_accepted_submissions
 from(select
		v.view_id,
		v.sum_total_views,
		v.sum_total_unique_views,
		s.sub_id,
		s.sum_total_submissions,
		s.sum_total_accepted_submissions
	from (select
			challenge_id as view_id,
			sum(total_views) as sum_total_views,
			sum(total_unique_views) as sum_total_unique_views
		from View_Stats
		group by challenge_id)	as v
	left join (select
				challenge_id as sub_id,
				sum(total_submissions) as sum_total_submissions,
				sum(total_accepted_submissions) as sum_total_accepted_submissions
			from Submission_Stats
			group by challenge_id) as s on s.sub_id = v.view_id
	UNION
	select
		v.view_id,
		v.sum_total_views,
		v.sum_total_unique_views,
		s.sub_id,
		s.sum_total_submissions,
		s.sum_total_accepted_submissions
	from (select
			challenge_id as view_id,
			sum(total_views) as sum_total_views,
			sum(total_unique_views) as sum_total_unique_views
		from View_Stats
		group by challenge_id)	as v
	right join (select
				challenge_id as sub_id,
				sum(total_submissions) as sum_total_submissions,
				sum(total_accepted_submissions) as sum_total_accepted_submissions
			from Submission_Stats
			group by challenge_id) as s on s.sub_id = v.view_id) as cont;


/*CONCATENANDO A TABELA DOS CHALLENGES COM AS SUBMISSIONS*/

select
	ident.concat_contest_id as contest_id,
	ident.hacker_id as hacker_id,
	ident.name as name,
	ident.chal_challenge_id as ident_challenge_id
from(select 
		chal.challenge_id as chal_challenge_id,
		chal.college_id as chal_college_id,
		concat.college_id as concat_college_id,
		concat.colle_contest_id as concat_contest_id,
		concat.hacker_id as hacker_id,
		concat.name as name	
	from (select
			colle.college_id as college_id,
			colle.contest_id as colle_contest_id,
			cont.contest_id as cont_contest_id,
			cont.hacker_id as hacker_id,
			cont.name as name
		from Colleges as colle
		left join Contests as cont on colle.contest_id = cont.contest_id) as concat
	left join Challenges as chal on concat.college_id = chal.college_id) as ident
left join (select
			v.view_id,
			v.sum_total_views,
			v.sum_total_unique_views,
			s.sub_id,
			s.sum_total_submissions,
			s.sum_total_accepted_submissions
		from (select
				challenge_id as view_id,
				sum(total_views) as sum_total_views,
				sum(total_unique_views) as sum_total_unique_views
			from View_Stats
			group by challenge_id)	as v
		left join (select
					challenge_id as sub_id,
					sum(total_submissions) as sum_total_submissions,
					sum(total_accepted_submissions) as sum_total_accepted_submissions
				from Submission_Stats
				group by challenge_id) as s on s.sub_id = v.view_id
		UNION
		select
			v.view_id,
			v.sum_total_views,
			v.sum_total_unique_views,
			s.sub_id,
			s.sum_total_submissions,
			s.sum_total_accepted_submissions
		from (select
				challenge_id as view_id,
				sum(total_views) as sum_total_views,
				sum(total_unique_views) as sum_total_unique_views
			from View_Stats
			group by challenge_id)	as v
		right join (select
					challenge_id as sub_id,
					sum(total_submissions) as sum_total_submissions,
					sum(total_accepted_submissions) as sum_total_accepted_submissions
				from Submission_Stats
				group by challenge_id) as s on s.sub_id = v.view_id) as counts
