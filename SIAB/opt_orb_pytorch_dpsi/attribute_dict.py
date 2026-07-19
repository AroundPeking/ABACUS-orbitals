class AttributeDict(dict):
	"""Minimal nested attribute dictionary used by the SIAB optimizer."""

	def __missing__(self, key):
		value = type(self)()
		self[key] = value
		return value

	def __getattr__(self, key):
		if key.startswith("__"):
			raise AttributeError(key)
		return self[key]

	def __setattr__(self, key, value):
		self[key] = value

