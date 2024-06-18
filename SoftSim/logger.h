#pragma once
#include "spdlog\spdlog.h"
#include <iostream>
#include <memory>
class vLogger {
private:
	vLogger();
	std::shared_ptr<spdlog::logger> spdlogger;
	static vLogger* logger;
public:
	static vLogger* getInstance();
	void logInfo(const char* info);
	void logWarnning(const char* info);
	void logError(const char* info);

	//输出格式化字符串(使用变长参数)
	//注意：模板不使用就不会被具体化，写在头文件中，不要写在
	template<typename Arg1, typename... Args>
	void logInfo(const char* infor, const Arg1 &arg1, const Args &... args) {
		getInstance()->spdlogger.get()->info(infor, arg1, args...);
	}

	template<typename Arg1, typename... Args>
	void logWarnning(const char* infor, const Arg1 &arg1, const Args &... args) {
		getInstance()->spdlogger.get()->warn(infor, arg1, args...);
	}

	template<typename Arg1, typename... Args>
	void logError(const char* infor, const Arg1 &arg1, const Args &... args) {
		getInstance()->spdlogger.get()->error(infor, arg1, args...);
	}
};