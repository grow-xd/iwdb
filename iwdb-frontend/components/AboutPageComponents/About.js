import { RiLinkM, RiLinksLine } from '@remixicon/react';
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
    return (
        <div className="min-h-[81.1vh] flex flex-col md:flex-row items-start w-full px-5 sm:px-10 md:px-20 lg:px-30 gap-5 md:mt-5">
            <div className="hidden md:sticky md:top-[8vh] h-auto w-full md:w-auto lg:w-auto md:flex flex-col gap-3  p-5 text-sm">
                <Link href="#making_the_internet_safer">
                    <div onClick={handleSectionOneClick} className={`${section === 1 && 'border-l-[3px] border-blue-500'} px-3`}>
                        <h1 className="">Making the Internet Safer</h1>
                    </div>
                </Link>
                <Link href="#how_it_works">
                    <div onClick={handleSectionTwoClick} className={`${section === 2 && 'border-l-[3px] border-blue-500'} px-3`}>
                        <h1 className="">How it works</h1>
                    </div>
                </Link>
                <Link href="#built_with">
                    <div onClick={handleSectionThreeClick} className={`${section === 3 && 'border-l-[3px] border-blue-500'} px-3`}>
                        <h1 className="">Built With</h1>
                    </div>
                </Link>
                <Link href="#our_vision">
                    <div onClick={handleSectionFourClick} className={`${section === 4 && 'border-l-[3px] border-blue-500'} px-3`}>
                        <h1 className="">Our Vision</h1>
                    </div>
                </Link>
            </div>
            <div className="h-auto rounded-[15px] w-full md:w-[70%] lg:w-[60%] flex flex-col gap-10 md:gap-20 border-[0.4px] border-neutral-400 dark:border-neutral-700 p-5 md:py-10 md:px-10">
                <div className="">
                    <Link href="#making_the_internet_safer">
                        <div onClick={handleSectionOneClick} id={"making_the_internet_safer"} className="flex gap-2 cursor-pointer group">
                            <h1 className="text-2xl font-bold">Making the Internet Safer — One Link at a Time</h1>
                            <div className="invisible group-hover:visible">
                                <RiLinkM/>
                            </div>
                        </div>
                    </Link>
                    <div className="mt-5">
                        <p className="">In an age where misleading websites and digital threats are just one click away, we&#39;re building a smarter way to surf the web. Our platform helps users verify the trustworthiness and safety of websites in real-time using AI and machine learning.</p>
                        <p className="mt-2">We combine automated web crawling, a browser extension for real-time checks, and powerful ML models to analyze links, assess risks, and deliver a transparent Trust Score for every website.</p>
                    </div>
                </div>
                <div className="">
                    <Link href="#how_it_works">
                        <div onClick={handleSectionTwoClick} id={"how_it_works"} className="flex gap-2 cursor-pointer group">
                            <h1 className="text-2xl font-bold">How It Works ??</h1>
                            <div className="invisible group-hover:visible">
                                <RiLinkM/>
                            </div>
                        </div>
                    </Link>
                    <div className="flex flex-col gap-5 mt-5">
                        <div className="flex gap-2">
                            <div className="text-lg font-semibold">1.</div>
                            <div className="flex flex-col gap-3">
                                <h1 className="text-lg font-semibold">Link Crawling</h1>
                                <p className="">Our system crawls submitted or detected links and gathers all necessary metadata, redirection chains, and content structure.</p>
                            </div>
                        </div>
                        <div className="flex gap-2">
                            <div className="text-lg font-semibold">2.</div>
                            <div className="flex flex-col gap-3">
                                <h1 className="text-lg font-semibold">AI-Based Analysis</h1>
                                <div className="">
                                    <p className="">The data is processed through our trained ML models that detect:</p>
                                    <ul className="list-disc ml-5 mt-2">
                                        <li className="">Phishing indicators</li>
                                        <li className="">Unsafe redirects</li>
                                        <li className="">Fake login pages</li>
                                        <li className="">Spam and suspicious content</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        <div className="flex gap-2">
                            <div className="text-lg font-semibold">3.</div>
                            <div className="flex flex-col gap-3">
                                <h1 className="text-lg font-semibold">Trust Score Generation</h1>
                                <p className="">Each link receives a score based on its safety, credibility, and potential risk.</p>
                            </div>
                        </div>
                        <div className="flex gap-2">
                            <div className="text-lg font-semibold">4.</div>
                            <div className="flex flex-col gap-3">
                                <h1 className="text-lg font-semibold">Public Link Database</h1>
                                <p className="">Previously crawled links are stored and indexed for quick re-checks by anyone visiting our website.</p>
                            </div>
                        </div>
                    </div>
                </div>
                <div className="">
                    <Link href="#built_with">
                        <div onClick={handleSectionThreeClick} id={"built_with"} className="flex gap-2 cursor-pointer group">
                            <h1 className="text-2xl font-bold">Built With</h1>
                            <div className="invisible group-hover:visible">
                                <RiLinkM/>
                            </div>
                        </div>
                    </Link>
                    <div className="mt-5">
                        <ul className="list-disc ml-5">
                            <li className=""><span className="font-semibold">Python</span> for backend & crawling logic</li>
                            <li className=""><span className="font-semibold">AWS S3</span> for secure, scalable storage</li>
                            <li className=""><span className="font-semibold">Scikit-learn & NLP</span> models for inference</li>
                            <li className=""><span className="font-semibold">Next.js</span> for a snappy, modern frontend</li>
                            <li className=""><span className="font-semibold">Browser extension</span> for real-time link checks</li>
                        </ul>
                    </div>
                </div>
                <div className="">
                    <Link href="#our_vision">
                        <div onClick={handleSectionFourClick} id={"our_vision"} className="flex gap-2 cursor-pointer group">
                            <h1 className="text-2xl font-bold">Our Vision</h1>
                            <div className="invisible group-hover:visible">
                                <RiLinkM/>
                            </div>
                        </div>
                    </Link>
                    <div className="mt-5">
                        <p className="">We believe web security should be transparent, accessible, and user-driven. Whether you&#39;re a casual browser or a cybersecurity enthusiast, our platform gives you the tools to browse with confidence.</p>
                    </div>
                </div>
            </div>
        </div>
    );
}
