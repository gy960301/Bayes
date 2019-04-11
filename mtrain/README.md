# Loss Logger 

## Usage

Init `_MLOGS` folder befor logging loss function.
```python
    setup_log_folder_name()
```
This will create a folder `_MLOGS/yy-MM-DD-HH-MM` and all logs will stored to this folder.

If you want change `_MLOGS` paht, just set `setup_log_folder_name` to dir you want.

```python
    dir = 'what you want'
    setup_log_folder_name(base_folder_name=dir)
```



Then initialize a `LLgoer` with `loss_name` 
```python
    loss_name = 'Entorpy_Loss'
    log = LLoger(loss_name)
    log.loss(5, 5.0)
```
Then contains in `Entropy_Loss.log`
```
2018-11-27 09:52:21,707 INFO [ADAM at step 5 is 5.0]
```

- add read log function
- function with `Visdom`