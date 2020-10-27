SELECT pname, min(evaluation.rating) as min_rating , max(evaluation.rating) as max_rating , 

avg(evaluation.rating) as avg_rating ,count(evaluation.rating) as no_rating FROM evaluation ,product 
where  product.p_id = evaluation.p_id 

group by pname