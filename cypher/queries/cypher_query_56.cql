#To asses and valdiate and check Sherpa paths in hypothesis space

"""  
match(n)-[r]-(m) where "sherpa" in r.source and toLower(n.name) contains "inflammation" and toLower(m.name) contains "ifng" return n,r,m
"""

"""

"""#To asses and valdiate and check  CBM  paths in hypothesis space

"""  
match(n)-[r]-(m) where  toLower(r.source) contains "cbm" and toLower(n.name) contains "inflammation" and toLower(m.name) contains "ifng" return n,r,m
"""