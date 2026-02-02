_T='⚠️ Erreur d’envoi :'
_S='text/plain'
_R='Content-Type'
_Q='integrity'
_P='answers'
_O='event_type'
_N='timestamp'
_M='quiz_title'
_L='student'
_K='notebook_id'
_J='session_hash'
_I='__code__'
_H='full_hash'
_G='__func__'
_F='retries'
_E='test_mode'
_D='exam_mode'
_C=None
_B='.'
_A='utf-8'
import base64,sys,hashlib,inspect,re,types
from importlib.metadata import metadata
from pathlib import Path
import requests,json,datetime,asyncio
def getUser():
	import os
	try:user=os.getlogin()
	except:user='erreurUser'
	return user
def compute_machine_id():import platform,hashlib;raw='|'.join([platform.node(),platform.system(),platform.release(),platform.machine(),platform.processor(),getUser()]);return hashlib.sha256(raw.encode()).hexdigest()[:16]
def get_source_integrity_hash(cls):
	try:source=inspect.getsource(cls);source=re.sub('#.*','',source);source='\n'.join([line.strip()for line in source.splitlines()if line.strip()]);return hashlib.sha256(source.encode(_A)).hexdigest()
	except(OSError,TypeError):return'Source non disponible'
def get_ultra_integrity_hash(cls):
	method_hashes=[]
	for(name,attr)in cls.__dict__.items():
		func=_C
		if inspect.isfunction(attr):func=attr
		elif isinstance(attr,(staticmethod,classmethod)):func=attr.__func__
		if func and hasattr(func,_I):code_obj=func.__code__;bytecode=code_obj.co_code;consts=str(code_obj.co_consts).encode(_A);signature=f"{name}:".encode(_A)+bytecode+consts;method_hashes.append(hashlib.sha256(signature).hexdigest())
	if not method_hashes:return
	method_hashes.sort();final_payload=''.join(method_hashes).encode(_A);return hashlib.sha256(final_payload).hexdigest()
def get_package_directory(package_name):
	import importlib.util;spec=importlib.util.find_spec(package_name)
	if spec is _C:raise ImportError(f"Le package {package_name} n'est pas disponible.")
	return Path(spec.origin).parent
def get_package_hash(package_name):
	from importlib.metadata import metadata;meta=metadata(package_name);keywords=meta.get_all('Keywords',[]);kz=keywords[0]
	for k in kz.split(','):
		if k.startswith('hash:'):return k.split(':',1)[1]
def package_hash(package_dir,exclude=_C,algo='sha256'):
	package_dir=Path(package_dir)if isinstance(package_dir,str)else package_dir;h=hashlib.new(algo);exclude=exclude or set();files=sorted(p for p in package_dir.rglob('*')if p.is_file()and p.name not in exclude and not any(ex in p.parts for ex in exclude))
	for path in files:h.update(path.read_bytes())
	return h.hexdigest(),files
def compute_local_hash(base_dir):
	h=hashlib.sha256()
	for p in sorted(base_dir.rglob('*.py')):h.update(p.read_bytes())
	return h.hexdigest()
def hash_fonction(func):
	h=hashlib.sha256()
	try:src=inspect.getsource(func);h.update(src.encode(_A))
	except(OSError,IOError):pass
	return h.digest()
def hash_fonction_new(func):
	h=hashlib.sha256();func=getattr(func,_G,func)
	if hasattr(func,_I):code=func.__code__;h.update(code.co_code);h.update(repr(code.co_consts).encode());h.update(repr(code.co_names).encode());h.update(repr(code.co_varnames).encode());h.update(repr(code.co_freevars).encode());h.update(repr(code.co_cellvars).encode());h.update(str(code.co_argcount).encode());h.update(str(code.co_kwonlyargcount).encode());h.update(str(code.co_flags).encode());h.update(code.co_filename.encode());h.update(str(code.co_firstlineno).encode())
	else:h.update(repr(func).encode())
	return h.digest()
def fonctions_premier_niveau(module):
	for(name,obj)in vars(module).items():
		if inspect.isfunction(obj)and(obj.__module__==module.__name__ or obj.__module__=='__main__'):yield(name,obj)
def methodes_de_classe_old(cls):
	for(name,obj)in cls.__dict__.items():
		if isinstance(obj,(staticmethod,classmethod)):yield(name,obj.__func__)
		elif inspect.isfunction(obj):yield(name,obj)
def methodes_de_classe(cls):
	for(name,obj)in cls.__dict__.items():
		if isinstance(obj,(staticmethod,classmethod)):func=obj.__func__
		elif inspect.isfunction(obj):func=obj
		else:continue
		if not func.__qualname__.startswith(cls.__name__+_B):continue
		yield(name,func)
def methodes_de_classe_plus(cls):
	for(name,obj)in cls.__dict__.items():
		if isinstance(obj,(staticmethod,classmethod)):func=obj.__func__
		elif inspect.isfunction(obj):func=obj
		else:continue
		if func.__module__!=cls.__module__:continue
		if not func.__qualname__.startswith(cls.__name__+_B):continue
		yield(name,func)
def hash_classe(cls):
	h=hashlib.sha256();h.update(cls.__name__.encode(_A))
	for base in cls.__bases__:h.update(base.__name__.encode(_A))
	for(name,method)in sorted(methodes_de_classe(cls),key=lambda x:x[0]):h.update(name.encode(_A));h.update(hash_fonction(method))
	return h.digest()
def classes_du_module(module):
	for(name,obj)in vars(module).items():
		if inspect.isclass(obj)and obj.__module__==module.__name__:yield(name,obj)
def hash_module(module):
	h=hashlib.sha256()
	for(name,func)in sorted(fonctions_premier_niveau(module),key=lambda x:x[0]):h.update(b'F');h.update(name.encode(_A));h.update(hash_fonction(func))
	for(name,cls)in sorted(classes_du_module(module),key=lambda x:x[0]):h.update(b'C');h.update(name.encode(_A));h.update(hash_classe(cls))
	return h.hexdigest()
def hash_methodes_vivantes(obj):
	h=hashlib.sha256()
	for(name,func)in sorted(methodes_vivantes_objet(obj)):h.update(b'M');h.update(name.encode());h.update(hash_fonction(func))
	return h.hexdigest()
def methodes_vivantes_objet(obj):
	cls=obj.__class__;module_name=cls.__module__
	for name in dir(obj):
		if name.startswith('__'):continue
		try:attr=getattr(obj,name)
		except Exception:continue
		if not callable(attr):continue
		func=getattr(attr,_G,attr)
		if func.__module__!=module_name:continue
		if not func.__qualname__.startswith(cls.__name__+_B):continue
		yield(name,func)
def hash_classe_vivante(cls):
	h=hashlib.sha256()
	for name in dir(cls):
		if name.startswith('__'):continue
		try:attr=getattr(cls,name)
		except Exception:continue
		if not callable(attr):continue
		func=getattr(attr,_G,attr)
		if func.__module__!=cls.__module__:continue
		if not func.__qualname__.startswith(cls.__name__+_B):continue
		h.update(name.encode());h.update(hash_fonction(func))
	return h.digest()
def hash_dependances_modules(obj,modules_cibles):
	cls=obj.__class__;module_name=cls.__module__;package=obj.__module__.split(_B,1)[0];h=hashlib.sha256();dependances=set()
	for(name,func)in methodes_vivantes_objet(obj):dependances|=set(func.__code__.co_names)
	for module in modules_cibles:
		for name in sorted(dependances):
			if not hasattr(module,name):continue
			obj=getattr(module,name)
			if isinstance(obj,(types.FunctionType,types.MethodType)):
				if obj.__module__!=module.__name__:continue
				h.update(hash_fonction(obj))
			else:0
	return h.hexdigest()
def stable_value(val):
	if val is _C or isinstance(val,(int,float,bool,str)):return repr(val)
	if isinstance(val,(list,tuple)):return'['+','.join(stable_value(v)for v in val)+']'
	if isinstance(val,dict):return'{'+','.join(f"{stable_value(k)}:{stable_value(v)}"for(k,v)in sorted(val.items(),key=lambda x:repr(x[0])))+'}'
	return f"<UNHASHABLE:{type(val).__name__}>"
def hash_watchlist(obj,WATCHLIST=[]):
	cfg_parts=[]
	for name in WATCHLIST:
		if hasattr(obj,name):val_str=stable_value(getattr(obj,name));cfg_parts.append(f"{name}:{val_str}")
	cfg_parts.sort();payload='|'.join(cfg_parts);return hashlib.sha256(payload.encode()).hexdigest()
def get_big_integrity_hash(obj,modules=[],WATCHLIST=[]):
	modules_hash=[];package=obj.__module__.split(_B,1)[0];modules=[sys.modules[package+_B+mod]for mod in modules]
	for module in modules:mh=hash_module(module);modules_hash.append(mh)
	watchlist_hash=hash_watchlist(obj,WATCHLIST);living_object=hash_methodes_vivantes(obj);dependances=hash_dependances_modules(obj,modules);payload_parts=modules_hash.copy();payload_parts.append(watchlist_hash);payload_parts.append(living_object);payload_parts.append(dependances);payload='|'.join(payload_parts);full_hash=hashlib.sha256(payload.encode()).hexdigest();return{'source_hash':modules_hash,'watchlist_hash':watchlist_hash,'live_object':living_object,'dependances':dependances,_H:full_hash}
def get_full_object_hash(obj,modules=[],WATCHLIST=[]):modules_hash=[];package=obj.__module__.split(_B,1)[0];modules=[sys.modules[package+_B+mod]for mod in modules];watchlist_hash=hash_watchlist(obj,WATCHLIST);living_object=hash_methodes_vivantes(obj);payload_parts=[];payload_parts.append(watchlist_hash);payload_parts.append(living_object);payload='|'.join(payload_parts);full_hash=hashlib.sha256(payload.encode()).hexdigest();return full_hash
async def do_the_check(obj,WORKDIR):
	B='utils';A='main';import shutil
	if not obj.stop_event.is_set():obj.stop_event.set()
	session_hash=compute_local_hash(WORKDIR);shutil.rmtree(WORKDIR)
	while not obj.stop_check_event.is_set():
		parameters={_D:obj.exam_mode,_E:obj.test_mode,_F:obj.retries,'counts':','.join(str(v)for(k,v)in sorted(obj.quiz_counts.items())),'corrections':','.join(str(v)for(k,v)in sorted(obj.quiz_correct.items())),_H:get_full_object_hash(obj,modules=[A,B],WATCHLIST=[_D,_E,_F])};big_hash=get_big_integrity_hash(obj,modules=[A,B],WATCHLIST=[_D,_E,_F]);parameters['get_big_integrity_hash']=big_hash;parameters[_J]=session_hash;payload={_K:obj.machine_id,_L:obj.student.name,_M:_Q,_N:datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),_O:'z_check_integrity','parameters':parameters,_P:{},'score':str(0)}
		try:r=requests.post(obj.SHEET_URL,data=json.dumps(payload),headers={_R:_S})
		except Exception as e:print(_T,e)
		try:await asyncio.wait_for(obj.stop_check_event.wait(),timeout=obj._CHECKALIVE)
		except asyncio.TimeoutError:pass
def do_the_check_old(obj,WORKDIR):
	session_hash=compute_local_hash(WORKDIR);EXCLUDE={'putils.py','__pycache__','.ipynb_checkpoints','.DS_Store'};labdir=get_package_directory('labquiz');installed_hash,f=package_hash(labdir,exclude=EXCLUDE);output={'installed_hash':installed_hash,_J:session_hash,_H:get_full_object_hash(obj),'src_hash':get_source_integrity_hash(obj.__class__),_F:obj.retries,_D:obj.exam_mode,_E:obj.test_mode,'transfer':obj.sheetTransfer,'quizfile':obj.QUIZFILE};payload={_K:obj.machine_id,_L:obj.student.name,_M:_Q,_N:datetime.datetime.now().isoformat(timespec='seconds'),_O:'teacher_check',_P:output,'score':0}
	try:r=requests.post(obj.SHEET_URL,data=json.dumps(payload),headers={_R:_S})
	except Exception as e:print(_T,e)
# noqa: E501  # 35e5c5e979187d19
