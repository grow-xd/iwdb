import { RiLinkM, RiLinksLine } from '@remixicon/react';
import Image from 'next/image';
import Link from 'next/link';
import React, {useState} from 'react'

export default function Guide() {
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
    const handleSectionFiveClick = () => {
        setSection(5);
    }
    return (
        <div className="min-h-[81.1vh] flex flex-col md:flex-row items-start w-full px-5 sm:px-10 md:px-20 lg:px-30 gap-5 md:mt-5">
            <div className="hidden md:sticky md:top-[8vh] h-auto w-full md:w-auto lg:w-auto md:flex flex-col gap-3  p-5 text-sm">
                <Link href="#using_the_website">
                    <div onClick={handleSectionOneClick} className={`${section === 1 && 'border-l-[3px] border-blue-500'} px-3`}>
                        <h1 className="">Using the Website</h1>
                    </div>
                </Link>
                <Link href="#browser_extension_guide">
                    <div onClick={handleSectionTwoClick} className={`${section === 2 && 'border-l-[3px] border-blue-500'} px-3`}>
                        <h1 className="">Browser Extension Guide</h1>
                    </div>
                </Link>
                <Link href="#how_the_crawler_and_scoring_works">
                    <div onClick={handleSectionThreeClick} className={`${section === 3 && 'border-l-[3px] border-blue-500'} px-3`}>
                        <h1 className="">How the Crawling & Scoring Works</h1>
                    </div>
                </Link>
                <Link href="#submitting_link_for_crawling">
                    <div onClick={handleSectionFourClick} className={`${section === 4 && 'border-l-[3px] border-blue-500'} px-3`}>
                        <h1 className="">Submitting a Link for Crawling</h1>
                    </div>
                </Link>
                <Link href="#tips_for_safer_browsing">
                    <div onClick={handleSectionFiveClick} className={`${section === 5 && 'border-l-[3px] border-blue-500'} px-3`}>
                        <h1 className="">Tips for Safer Browsing</h1>
                    </div>
                </Link>
            </div>
            <div className="h-auto rounded-[15px] w-full md:w-[70%] lg:w-[60%] flex flex-col gap-10 md:gap-20 border-[0.4px] border-neutral-400 dark:border-neutral-700 p-5 md:py-10 md:px-10">
                <div className="">
                    <Link href="#using_the_website">
                        <div onClick={handleSectionOneClick} id={"using_the_website"} className="flex gap-2 cursor-pointer group">
                            <h1 className="text-2xl font-bold">Using the Website</h1>
                            <div className="invisible group-hover:visible">
                                <RiLinkM/>
                            </div>
                        </div>
                    </Link>
                    <div className="mt-5 flex flex-col gap-3">
                        <p className="">Our website is your dashboard for verifying link safety. Here&#39;s how to use it:</p>
                        <ul className="list-disc ml-5">
                            <li className=""><span className="font-semibold">Search a Link:</span> Paste any URL into the search bar to check if it has been previously crawled and rated.</li>
                            <li className=""><span className="font-semibold">Trust Score:</span> View the ML-generated trust score for the link, along with indicators (e.g., phishing risk, redirects, domain age).</li>
                            <li className=""><span className="font-semibold">Link History:</span> Browse recent or popular searches to see how other users are interacting with links.</li>
                        </ul>
                        <p className="italic">If a link hasn&#39;t been crawled yet, we&#39;ll queue it for crawling and notify you when it&#39;s ready.</p>
                    </div>
                </div>
                <div className="">
                    <Link href="#browser_extension_guide">
                        <div onClick={handleSectionTwoClick} id={"browser_extension_guide"} className="flex gap-2 cursor-pointer group">
                            <h1 className="text-2xl font-bold">Browser Extension Guide</h1>
                            <div className="invisible group-hover:visible">
                                <RiLinkM/>
                            </div>
                        </div>
                    </Link>
                    <div className="mt-5 flex flex-col gap-3">
                        <p className="">Install our extension to get real-time link analysis while browsing.</p>
                        <ul className="list-disc ml-5">
                            <li className=""><span className="font-semibold">Download the Extension</span> from our website (or Chrome Web Store).</li>
                            <li className=""><span className="font-semibold">Pin it</span> to your toolbar for easy access.</li>
                            <li className=""><span className="font-semibold">Click any link</span> and the extension will instantly show the trust score.</li>
                            <li className=""><span className="font-semibold">Report Suspicious Links</span> directly through the extension interface.</li>
                        </ul>
                        <p className="italic">We respect your privacy. No personal data is collected.</p>
                    </div>
                </div>
                <div className="">
                    <Link href="#how_the_crawler_and_scoring_works">
                        <div onClick={handleSectionThreeClick} id={"how_the_crawler_and_scoring_works"} className="flex gap-2 cursor-pointer group">
                            <h1 className="text-2xl font-bold">How the Crawling & Scoring Works</h1>
                            <div className="invisible group-hover:visible">
                                <RiLinkM/>
                            </div>
                        </div>
                    </Link>
                    <div className="mt-5 flex flex-col gap-3">
                        <p className="">Behind the scenes, here&#39;s what happens:</p>
                        <ul className="list-disc ml-5">
                            <li className="">Our <span className="font-semibold">crawler</span> fetches and processes the content of the link.</li>
                            <li className="">An <span className="font-semibold">ML model</span> analyzes elements such as:</li>
                            <li className="">The system generates a <span className="font-semibold">Trust Score (0-100)</span>, color-coded for clarity:</li>
                                <ul className=" ml-4 list-disc text-sm my-1">
                                    <li className="">
                                        <div className="flex items-center gap-2">
                                            <div className="h-4 w-4 bg-green-500 rounded-[5px]"></div>
                                            <h1 className=""> 80-100: Safe</h1>
                                        </div>
                                    </li>
                                    <li className="">
                                        <div className="flex items-center gap-2">
                                            <div className="h-4 w-4 bg-amber-400 rounded-[5px]"></div>
                                            <h1 className=""> 50-79: Caution</h1>
                                        </div>
                                    </li>
                                    <li className="">
                                        <div className="flex items-center gap-2">
                                            <div className="h-4 w-4 bg-red-500 rounded-[5px]"></div>
                                            <h1 className=""> 0-49: Unsafe</h1>
                                        </div>
                                    </li>
                                </ul>
                            <li className="">An easy-to-understand visual summary of the site&#39;s trustworthiness through <span className="">rating (out of 5 stars)</span>, influenced by:</li>
                                <ul className=" ml-4 list-disc text-sm my-1">
                                    <li className="">Technical analysis</li>
                                    <li className="">User feedback</li>
                                    <li className="">Past crawl consistency</li>
                                    <li className="">Frequency of malicious reports</li>
                                </ul>
                            <li className="list-none">
                                <div className="flex flex-col gap-2">
                                    <h1 className="">Example:</h1>
                                    <div className="flex items-center gap-1">
                                        <div className="h-4 w-4">
                                            <Image src={"/star.png"} alt={"start"} width={1440} height={1440} />
                                        </div>
                                        <div className="h-4 w-4">
                                            <Image src={"/star.png"} alt={"start"} width={1440} height={1440} />
                                        </div>
                                        <div className="h-4 w-4">
                                            <Image src={"/star.png"} alt={"start"} width={1440} height={1440} />
                                        </div>
                                        <div className="h-4 w-4">
                                            <Image src={"/star.png"} alt={"start"} width={1440} height={1440} />
                                        </div>
                                        <div className="h-4 w-4">
                                            <Image src={"/star.png"} alt={"start"} width={1440} height={1440} />
                                        </div>
                                        <h1 className=""> - Highly trusted, consistent behaviour</h1>
                                    </div>
                                    <div className="flex items-center gap-1">
                                        <div className="h-4 w-4">
                                            <Image src={"/star.png"} alt={"start"} width={1440} height={1440} />
                                        </div>
                                        <div className="h-4 w-4">
                                            <Image src={"/star.png"} alt={"start"} width={1440} height={1440} />
                                        </div>
                                        <div className="h-4 w-4">
                                            <Image src={"/star.png"} alt={"start"} width={1440} height={1440} />
                                        </div>
                                        <h1 className=""> - Unreliable, warning signs present</h1>
                                    </div>
                                </div>
                            </li>
                            <li className="">Every site is ranked <span className="font-semibold"> relative to other crawled websites</span>, showing how it compares in terms of safety, reputation, and popularity. This is useful for identifying:</li>
                                <ul className=" ml-4 list-disc text-sm my-1">
                                    <li className="">Top trusted domains</li>
                                    <li className="">Frequently reported or flagged domains</li>
                                    <li className="">New or suspicious entries gaining traction</li>
                                </ul>
                        </ul>
                    </div>
                </div>
                <div className="">
                    <Link href="#submitting_link_for_crawling">
                        <div onClick={handleSectionFourClick} id={"submitting_link_for_crawling"} className="flex gap-2 cursor-pointer group">
                            <h1 className="text-2xl font-bold">Submitting a Link for Crawling</h1>
                            <div className="invisible group-hover:visible">
                                <RiLinkM/>
                            </div>
                        </div>
                    </Link>
                    <div className="mt-5 flex flex-col gap-3">
                        <p className="">You can also manually submit links that aren&#39;t yet rated.</p>
                        <ul className="list-disc ml-5">
                            <li className="">Go to the <span className="font-semibold">&#34;Submit a Link&#34;</span> section</li>
                            <li className="">Enter the URL</li>
                            <li className="">Hit <span className="font-semibold">Submit</span></li>
                            <li className="">Our system will crawl and notify you once complete</li>
                        </ul>
                        <p className="italic">Usually takes 2-5 minutes depending on the depth of the site.</p>
                    </div>
                </div>
                <div className="">
                    <Link href="#tips_for_safer_browsing">
                        <div onClick={handleSectionFiveClick} id={"tips_for_safer_browsing"} className="flex gap-2 cursor-pointer group">
                            <h1 className="text-2xl font-bold">Tips for Safer Browsing</h1>
                            <div className="invisible group-hover:visible">
                                <RiLinkM/>
                            </div>
                        </div>
                    </Link>
                    <div className="mt-5 flex flex-col gap-3">
                        <ul className="list-disc ml-5">
                            <li className="">Avoid clicking unknown or shortened links.</li>
                            <li className="">Use our extension on emails, forums, and social media platforms.</li>
                            <li className="">Check Trust Score before entering personal information on a site.</li>
                            <li className="">Report suspicious or fake websites — help the community stay safe.</li>
                        </ul>
                    </div>
                </div>
                <div className="">
                    <Link href="#tips_for_safer_browsing">
                        <div onClick={handleSectionFiveClick} id={"tips_for_safer_browsing"} className="flex gap-2 cursor-pointer group">
                            <h1 className="text-2xl font-bold">Still Need Help ?</h1>
                            <div className="invisible group-hover:visible">
                                <RiLinkM/>
                            </div>
                        </div>
                    </Link>
                    <div className="mt-5 flex flex-col gap-3">
                        <p className="">Visit our <Link href="/support" className="text-blue-500 dark:text-blue-400">Support Page</Link> or contact us at <Link href="mailto:support@iwdb.in" className="text-blue-500 dark:text-blue-400">support@iwdb.in</Link></p>
                        <p className="-mt-2">We&#34;re always happy to assist you!</p>
                    </div>
                </div>
            </div>
        </div>
    );
}
