select * from sonntagsfrage.results_questionaire;


-- Create script fuer Tabelle sonntagsfrage.results_questionaire
USE [sonntagsfrage-sql-db]
GO

/****** Objekt: Table [sonntagsfrage].[results_questionaire] Skriptdatum: 30.10.2020 17:12:07 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

select count(*) from [sonntagsfrage].[results_questionaire]

CREATE TABLE [sonntagsfrage].[results_questionaire] (
    [Datum]          VARCHAR (100) NULL,
    [CDU_CSU]        VARCHAR (100) NULL,
    [SPD]            VARCHAR (100) NULL,
    [GRUENE]         VARCHAR (100) NULL,
    [FDP]            VARCHAR (100) NULL,
    [LINKE]          VARCHAR (100) NULL,
    [PIRATEN]        VARCHAR (100) NULL,
    [AfD]            VARCHAR (100) NULL,
    [Linke_PDS]      VARCHAR (100) NULL,
    [PDS]            VARCHAR (100) NULL,
    [REP_DVU]        VARCHAR (100) NULL,
    [Sonstige]       VARCHAR (100) NULL,
    [Befragte]       VARCHAR (100) NULL,
    [Zeitraum]       VARCHAR (100) NULL,
    [meta_insert_ts] VARCHAR (100) NULL
);
GO

drop table [sonntagsfrage].[results_questionaire_clean];
CREATE TABLE [sonntagsfrage].[results_questionaire_clean] (
    [Datum]          VARCHAR (100) NULL,
    [CDU_CSU]        numeric NULL default 0,
    [SPD]            numeric NULL default 0,
    [GRUENE]         numeric NULL default 0,
    [FDP]            numeric NULL default 0,
    [LINKE]          numeric NULL default 0,
    [PIRATEN]        numeric NULL default 0,
    [AfD]            numeric NULL default 0,
    [Linke_PDS]      numeric NULL default 0,
    [PDS]            numeric NULL default 0,
    [REP_DVU]        numeric NULL default 0,
    [Sonstige]       numeric NULL default 0,
    [Befragte]       VARCHAR (100) NULL,
    [Zeitraum]       VARCHAR (100) NULL,
    [meta_insert_ts] VARCHAR (100) NULL
);
GO

update [sonntagsfrage].[results_questionaire_clean]
set meta_insert_ts= '@'
where meta_insert_ts= ''
;

select * from [sonntagsfrage].[results_questionaire_clean]
where Datum = ''

select * from [sonntagsfrage].[results_questionaire_clean]
where Befragte = ''


ALTER TABLE [sonntagsfrage].[results_questionaire_clean] drop CONSTRAINT DF_SomeNam2 ;
ALTER TABLE [sonntagsfrage].[results_questionaire_clean] ADD CONSTRAINT DF_SomeName DEFAULT '@' FOR Datum;
ALTER TABLE [sonntagsfrage].[results_questionaire_clean] ADD CONSTRAINT DF_SomeName2 DEFAULT '@' FOR Befragte;
ALTER TABLE [sonntagsfrage].[results_questionaire_clean] ADD CONSTRAINT DF_SomeName3 DEFAULT '@' FOR Zeitraum;
ALTER TABLE [sonntagsfrage].[results_questionaire_clean] ADD CONSTRAINT DF_SomeName4 DEFAULT '@' FOR meta_insert_ts;

truncate table sonntagsfrage.results_questionaire_clean;
insert into sonntagsfrage.results_questionaire_clean
select  
Datum
,cast( coalesce( '0', replace(isnull(CDU_CSU  , '0'), '-', '0')) as numeric) CDU_CSU
,cast( coalesce( '0', replace(isnull(SPD      , '0'), '-', '0')) as numeric) SPD          
,cast( coalesce( '0', replace(isnull(GRUENE   , '0'), '-', '0')) as numeric) GRUENE      
,cast( coalesce( '0', replace(isnull(FDP      , '0'), '-', '0')) as numeric) FDP        
,cast( coalesce( '0', replace(isnull(LINKE    , '0'), '-', '0')) as numeric) LINKE        
,cast( coalesce( '0', replace(isnull(PIRATEN  , '0'), '-', '0')) as numeric) PIRATEN      
,cast( coalesce( '0', replace(isnull(AfD      , '0'), '-', '0')) as numeric) AfD       
,cast( coalesce( '0', replace(isnull(Linke_PDS, '0'), '-', '0')) as numeric) Linke_PDS    
,cast( coalesce( '0', replace(isnull(PDS      , '0'), '-', '0')) as numeric) PDS          
,cast( coalesce( '0', replace(isnull(REP_DVU  , '0'), '-', '0')) as numeric) REP_DVU  
, cast( coalesce( '0', replace(replace(replace( isnull(Sonstige , '0'), '-', '0'), 'PIR', '0'), 'WASG3', '0')) as numeric) Sonstige
,Befragte   
,Zeitraum
,''
-- into sonntagsfrage.test 
from sonntagsfrage.results_questionaire q
GO

select REP_DVU
, cast( coalesce( '0', replace('', '', '0') ) as numeric)
,replace(replace(isnull(REP_DVU  , '0'), '-', '0'),'.', ',') REP_DVU  
--,cast(replace(replace(isnull(REP_DVU  , '0'), '-', '0'),'.', ',') as numeric)  REP_DVU  
from sonntagsfrage.results_questionaire q


select * from sonntagsfrage.results_questionaire
select * from sonntagsfrage.results_questionaire_clean

 select * from sonntagsfrage.results_questionaire_clean
        union all
        select '10.11.2020', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '0', '0', '0'




-- handle predictions
drop table [sonntagsfrage].[results_questionaire_clean];
CREATE TABLE [sonntagsfrage].[predictions_questionaire](
    [Datum]          VARCHAR (100) NULL default '@',
    [CDU_CSU]        numeric NULL default 0,
    [SPD]            numeric NULL default 0,
    [GRUENE]         numeric NULL default 0,
    [FDP]            numeric NULL default 0,
    [LINKE]          numeric NULL default 0,
    [PIRATEN]        numeric NULL default 0,
    [AfD]            numeric NULL default 0,
    [Linke_PDS]      numeric NULL default 0,
    [PDS]            numeric NULL default 0,
    [REP_DVU]        numeric NULL default 0,
    [Sonstige]       numeric NULL default 0,
    [Befragte]       VARCHAR (100) NULL default '@',
    [Zeitraum]       VARCHAR (100) NULL default '@',
    [meta_insert_ts] VARCHAR (100) NULL default '@'
);
GO

ALTER TABLE [sonntagsfrage].[predictions_questionaire] drop constraint DF_SomeNam;
ALTER TABLE [sonntagsfrage].[predictions_questionaire] drop constraint DF_SomeNam2;
ALTER TABLE [sonntagsfrage].[predictions_questionaire] drop constraint DF_SomeNam3;
ALTER TABLE [sonntagsfrage].[predictions_questionaire] drop constraint DF_SomeNam4;
ALTER TABLE [sonntagsfrage].[predictions_questionaire] drop column meta_insert_ts;
ALTER TABLE [sonntagsfrage].[predictions_questionaire] drop column befragte;
ALTER TABLE [sonntagsfrage].[predictions_questionaire] drop column zeitraum;

update [sonntagsfrage].[predictions_questionaire]
set [Datum]= '@'
where [Datum] is null
;

select * from [sonntagsfrage].[results_questionaire_clean];
select * from [sonntagsfrage].[predictions_questionaire];

ALTER TABLE [sonntagsfrage].[results_questionaire_clean] drop CONSTRAINT DF_SomeNam2 ;
ALTER TABLE [sonntagsfrage].[predictions_questionaire] ADD CONSTRAINT DF_SomeNam DEFAULT '@' FOR Datum;
ALTER TABLE [sonntagsfrage].[predictions_questionaire] ADD CONSTRAINT DF_SomeNam2 DEFAULT '@' FOR Befragte;
ALTER TABLE [sonntagsfrage].[predictions_questionaire] ADD CONSTRAINT DF_SomeNam3 DEFAULT '@' FOR Zeitraum;
ALTER TABLE [sonntagsfrage].[predictions_questionaire] ADD CONSTRAINT DF_SomeNam4 DEFAULT '@' FOR meta_insert_ts;


select count(*) from [sonntagsfrage].[predictions_questionaire]

select * from sonntagsfrage.predictions_questionaire p
select replace(r.Datum,'*','') as Datum_clean, r.* from sonntagsfrage.results_questionaire_clean r 
where substring(Datum, 3,1) = '.'
select convert(datetime,replace(r.Datum,'*',''), 104 ) as Datum_clean, r.* from sonntagsfrage.results_questionaire_clean r 


select * 
from sonntagsfrage.predictions_questionaire p
join sonntagsfrage.results_questionaire_clean r on p.Datum = r.Datum
select * from sonntagsfrage.results_questionaire_clean 

create or alter view [sonntagsfrage].[v_results_questionaire_clean_pivot] as 
select p.Datum, p.CDU_CSU, 'CDU_CSU' as Partei from [sonntagsfrage].[predictions_questionaire] p
union all select p.Datum, p.SPD, 'SPD' as Partei from [sonntagsfrage].[predictions_questionaire] p
union all select p.Datum, p.GRUENE, 'GRUENE' as Partei from [sonntagsfrage].[predictions_questionaire] p
union all select p.Datum, p.FDP, 'FDP' as Partei from [sonntagsfrage].[predictions_questionaire] p
union all select p.Datum, p.LINKE, 'LINKE' as Partei from [sonntagsfrage].[predictions_questionaire] p
union all select p.Datum, p.PIRATEN, 'PIRATEN' as Partei from [sonntagsfrage].[predictions_questionaire] p
union all select p.Datum, p.AfD, 'AfD' as Partei from [sonntagsfrage].[predictions_questionaire] p
union all select p.Datum, p.Linke_PDS, 'Linke_PDS' as Partei from [sonntagsfrage].[predictions_questionaire] p
union all select p.Datum, p.PDS, 'PDS' as Partei from [sonntagsfrage].[predictions_questionaire] p
union all select p.Datum, p.REP_DVU, 'REP_DVU' as Partei from [sonntagsfrage].[predictions_questionaire] p
union all select p.Datum, p.Sonstige, 'Sonstige' as Partei from [sonntagsfrage].[predictions_questionaire] p


select * from [sonntagsfrage].[v_predictions_questionaire_pivot]


create or alter view [sonntagsfrage].[v_results_questionaire_clean_pivot] as 
select p.Datum, p.CDU_CSU, 'CDU_CSU' as Partei                  , p.Befragte, p.Zeitraum, p.meta_insert_ts from [sonntagsfrage].[results_questionaire_clean] p
union all select p.Datum, p.SPD, 'SPD' as Partei                , p.Befragte, p.Zeitraum, p.meta_insert_ts from [sonntagsfrage].[results_questionaire_clean] p
union all select p.Datum, p.GRUENE, 'GRUENE' as Partei          , p.Befragte, p.Zeitraum, p.meta_insert_ts from [sonntagsfrage].[results_questionaire_clean] p
union all select p.Datum, p.FDP, 'FDP' as Partei                , p.Befragte, p.Zeitraum, p.meta_insert_ts from [sonntagsfrage].[results_questionaire_clean] p
union all select p.Datum, p.LINKE, 'LINKE' as Partei            , p.Befragte, p.Zeitraum, p.meta_insert_ts from [sonntagsfrage].[results_questionaire_clean] p
union all select p.Datum, p.PIRATEN, 'PIRATEN' as Partei        , p.Befragte, p.Zeitraum, p.meta_insert_ts from [sonntagsfrage].[results_questionaire_clean] p
union all select p.Datum, p.AfD, 'AfD' as Partei                , p.Befragte, p.Zeitraum, p.meta_insert_ts from [sonntagsfrage].[results_questionaire_clean] p
union all select p.Datum, p.Linke_PDS, 'Linke_PDS' as Partei    , p.Befragte, p.Zeitraum, p.meta_insert_ts from [sonntagsfrage].[results_questionaire_clean] p
union all select p.Datum, p.PDS, 'PDS' as Partei                , p.Befragte, p.Zeitraum, p.meta_insert_ts from [sonntagsfrage].[results_questionaire_clean] p
union all select p.Datum, p.REP_DVU, 'REP_DVU' as Partei        , p.Befragte, p.Zeitraum, p.meta_insert_ts from [sonntagsfrage].[results_questionaire_clean] p
union all select p.Datum, p.Sonstige, 'Sonstige' as Partei      , p.Befragte, p.Zeitraum, p.meta_insert_ts from [sonntagsfrage].[results_questionaire_clean] p

select * from [sonntagsfrage].[v_results_questionaire_clean_pivot]


