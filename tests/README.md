# Easy testing
Convenient alias to simplify the testing experience.
Add this to your `~/.bashrc`
```
testpy(){
    if [ -d "${PWD}/generalization/tests" ]; then
        python -m unittest discover generalization/tests/
    else
        echo "There is no tests directory in this path"
    fi 
}
```
