import { RiLinkM } from '@remixicon/react';
import Link from 'next/link';
import React, {useState} from 'react'

export default function About() {
    const [section, setSection] = useState(1);
    const handleSectionOneClick = () => {
        setSection(1);
    }
    const handleSectionTwoClick = () => {
        setSection(2);
    }
    const handleSectionThreeClick = () => {
        setSection(3);
    }
    const handleSectionFourClick = () => {
        setSection(4);
    }

    const handleSubmit = (e) => {
        e.preventDefault();
        // enter your support form submittion logic here...
    }

    return (
        <div className="min-h-[81.1vh] flex flex-col md:flex-row items-start w-full px-5 sm:px-10 md:px-20 lg:px-30 gap-5 md:mt-5">
            <div className="hidden md:sticky md:top-[8vh] h-auto w-full md:w-auto lg:w-auto md:flex flex-col gap-3  p-5 text-sm">
                <Link href="#we_are_here_to_help">
                    <div onClick={handleSectionOneClick} className={`${section === 1 && 'border-l-[3px] border-blue-500'} px-3`}>
                        <h1 className="">We&#39;re Here to Help</h1>
                    </div>
                </Link>
                <Link href="#common_issues_we_handle">
                    <div onClick={handleSectionTwoClick} className={`${section === 2 && 'border-l-[3px] border-blue-500'} px-3`}>
                        <h1 className="">Common Issues We Handle</h1>
                    </div>
                </Link>
                <Link href="#submit_request">
                    <div onClick={handleSectionThreeClick} className={`${section === 3 && 'border-l-[3px] border-blue-500'} px-3`}>
                        <h1 className="">Submit a Request</h1>
                    </div>
                </Link>
                <Link href="#still_need_help">
                    <div onClick={handleSectionFourClick} className={`${section === 4 && 'border-l-[3px] border-blue-500'} px-3`}>
                        <h1 className="">Still Need Help?</h1>
                    </div>
                </Link>
            </div>
            <div className="h-auto rounded-[15px] w-full md:w-[70%] lg:w-[60%] flex flex-col gap-10 md:gap-20 border-[0.4px] border-neutral-400 dark:border-neutral-700 p-5 md:py-10 md:px-10">
                <div className="">
                    <Link href="#we_are_here_to_help">
                        <div onClick={handleSectionOneClick} id={"we_are_here_to_help"} className="flex gap-2 cursor-pointer group">
                            <h1 className="text-2xl font-bold">We&#39;re Here to Help</h1>
                            <div className="invisible group-hover:visible">
                                <RiLinkM/>
                            </div>
                        </div>
                    </Link>
                    <div className="mt-5">
                        <p className="">Whether you&#39;re facing an issue, have a suggestion, or want to report a suspicious link — we&#39;re just one message away. Our team is committed to keeping your browsing experience safe, seamless, and informed.</p>
                    </div>
                </div>
                <div className="">
                    <Link href="#common_issues_we_handle">
                        <div onClick={handleSectionTwoClick} id={"common_issues_we_handle"} className="flex gap-2 cursor-pointer group">
                            <h1 className="text-2xl font-bold">Common Issues We Handle</h1>
                            <div className="invisible group-hover:visible">
                                <RiLinkM/>
                            </div>
                        </div>
                    </Link>
                    <div className="flex flex-col gap-2 mt-5">
                        <p className="">The data is processed through our trained ML models that detect:</p>
                        <ul className="list-disc ml-5">
                            <li className="">Phishing indicators</li>
                            <li className="">Unsafe redirects</li>
                            <li className="">Fake login pages</li>
                            <li className="">Spam and suspicious content</li>
                        </ul>
                    </div>
                </div>
                <div className="">
                    <Link href="#submit_request">
                        <div onClick={handleSectionThreeClick} id={"submit_request"} className="flex gap-2 cursor-pointer group">
                            <h1 className="text-2xl font-bold">Submit a Request</h1>
                            <div className="invisible group-hover:visible">
                                <RiLinkM/>
                            </div>
                        </div>
                    </Link>
                    <div className="flex flex-col gap-2 mt-5">
                        <p className="">The data is processed through our trained ML models that detect:</p>
                        <form onSubmit={handleSubmit} className="flex flex-col gap-5 mt-3">
                            <div className="flex flex-col gap-2">
                                <h1 className="text-neutral-500 dark:text-neutral-400">Full Name <span className="text-red-500">*</span></h1>
                                <input
                                    type=""
                                    placeholder="Enter your full name"
                                    className="w-full outline-none px-5 py-2 border border-neutral-400 dark:border-neutral-700 rounded-[10px]"
                                />
                            </div>
                            <div className="flex flex-col gap-2">
                                <h1 className="text-neutral-500 dark:text-neutral-400">Email <span className="text-red-500">*</span></h1>
                                <input
                                    type=""
                                    placeholder="Enter your email"
                                    className="w-full outline-none px-5 py-2 border border-neutral-400 dark:border-neutral-700 rounded-[10px]"
                                />
                            </div>
                            <div className="flex flex-col gap-2">
                                <h1 className="text-neutral-500 dark:text-neutral-400">Issue Type <span className="text-red-500">*</span></h1>
                                <div className="px-5 py-2 border border-neutral-400 dark:border-neutral-700 rounded-[10px]">
                                    <select className="w-full outline-none ">
                                        <option value="" className="">Select the Type of Issue</option>
                                        <option value="bug">Bug</option>
                                        <option value="feedback">Feedback</option>
                                        <option value="link report">Link Report</option>
                                        <option value="extension issue">Extension Issue</option>
                                        <option value="other">Other</option>
                                    </select>
                                </div>
                            </div>
                            <div className="flex flex-col gap-2">
                                <h1 className="text-neutral-500 dark:text-neutral-400">Description <span className="text-red-500">*</span></h1>
                                <textarea
                                    type=""
                                    placeholder="Write description here..."
                                    rows={2}
                                    className="resize-none w-full outline-none px-5 py-2 border border-neutral-400 dark:border-neutral-700 rounded-[10px]"
                                />
                            </div>
                            <div className="flex flex-col gap-2">
                                <h1 className="text-neutral-500 dark:text-neutral-400">Attachment (Optional):</h1>
                                <input
                                    type="file"
                                    placeholder=""
                                    className="w-full outline-none px-5 py-2 border border-neutral-400 dark:border-neutral-700 rounded-[10px] file:text-neutral-400"
                                />
                            </div>
                            <div className="mt-5 flex items-center justify-center md:justify-start">
                                <input
                                    type="submit"
                                    value="submit"
                                    className="px-5 py-2 bg-blue-500 text-sm text-white font-bold dark:bg-blue-600 rounded-[10px]"
                                />
                            </div>
                        </form>
                    </div>
                </div>
                <div className="">
                    <Link href="#still_need_help">
                        <div onClick={handleSectionFourClick} id={"still_need_help"} className="flex gap-2 cursor-pointer group">
                            <h1 className="text-2xl font-bold">Still need Help ?</h1>
                            <div className="invisible group-hover:visible">
                                <RiLinkM/>
                            </div>
                        </div>
                    </Link>
                    <div className="mt-3">
                        <p className=""><span className="font-bold">Email us</span>: <Link href="mailto:support@iwdb.in" className="text-blue-500 dark:text-blue-400">support@iwdb.in</Link></p>
                    </div>
                </div>
            </div>
        </div>
    );
}
