#include "logger.h"
vLogger* vLogger::logger = nullptr;

vLogger::vLogger() {
	//´´½¨console
	spdlogger = spdlog::stdout_color_mt("console");
}

vLogger* vLogger::getInstance() {
	if (logger == nullptr) {
		logger = new vLogger();
	}
	return logger;
}

void vLogger::logInfo(const char* info) {
	getInstance()->spdlogger.get()->info(info);
	
}
void vLogger::logWarnning(const char* info) {
	getInstance()->spdlogger.get()->warn(info);
}
void vLogger::logError(const char* info) {
	getInstance()->spdlogger.get()->error(info);
}

