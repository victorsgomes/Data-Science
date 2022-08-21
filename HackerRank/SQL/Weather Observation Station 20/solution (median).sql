SET @row_index := -1;

SELECT 
    avg(LAT_N) as median_value
FROM (
    SELECT @row_index:=@row_index + 1 AS row_index, score
    FROM scores
    ORDER BY score
  ) AS subq
  WHERE subq.row_index IN (FLOOR(@row_index / 2) , CEIL(@row_index / 2));