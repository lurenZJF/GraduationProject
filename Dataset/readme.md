+ 1.event_story.csv      
 
```
事件中文数据集
doc_id|story_id|event_id|category|time|keywords|main_keywords|ner|ner_keywords|title|content   

共有21708条推文，9943个event,2760个sotry
```



+ 2.TwitterEvent2012.npy   
  
```
事件英文数据集
["event_id", "tweet_id", "text", "created_at", "entity", "words"]
共有55943条数据
```
   
   
+ 3.save2mongodb.py  
将数据存储到mongodb数据库中，方便后续的时间分析
